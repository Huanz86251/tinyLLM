import math
import os
from os.path import split
from transformers import GenerationMixin, GenerationConfig, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
import copy
from typing import Dict, List, Optional, Literal,Union,Tuple,Any
from types import SimpleNamespace
from model.config import Config
from pytorch_tcn import TCN,TemporalConv1d
from datetime import datetime
import torch

import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.utils.checkpoint import checkpoint
from torch.nn import AdaptiveLogSoftmaxWithLoss
import math
import torch
try:
    from transformers.models.mamba2 import Mamba2Config, Mamba2Model
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
    HAS_MAMBA2 = True
except Exception as e:
    Mamba2Config = None
    Mamba2Model = None
    Mamba2Cache = None
    HAS_MAMBA2 = False
    warnings.warn(
        f"[TinyLLM] transformers.models.mamba2 未能导入（{e}）。"
        "SSM/Mamba2 将被禁用；请确保 cfg.use_ssm=False 或安装带 mamba2 支持的 transformers。",
        RuntimeWarning,
    )

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.modeling_rope_utils import rope_config_validation, ROPE_INIT_FUNCTIONS


def gumbel_noise(x:torch.Tensor):
    #-log(-log(U))
    u=torch.rand_like(x).clamp(1e-9,1-1e-9)
    return -torch.log(-torch.log(u))

def _build_mlp_ratio_schedule(cfg: Config) -> list[float]:
    L = int(cfg.num_hidden_layers)
    # 若未提供分界，则退回统一 mlp_ratio
    if cfg.mlp_mid_start is None and cfg.mlp_back_start is None:
        return [float(cfg.mlp_ratio)] * L

    mid = cfg.mlp_mid_start if cfg.mlp_mid_start is not None else L
    back = cfg.mlp_back_start if cfg.mlp_back_start is not None else L


    mid = max(0, min(int(mid), L))
    back = max(mid, min(int(back), L))

    ratios = (
        [float(cfg.mlp_ratio_front)] * mid +
        [float(cfg.mlp_ratio_mid)]   * (back - mid) +
        [float(cfg.mlp_ratio_back)]  * (L - back)
    )

    ov = getattr(cfg, "mlp_ratio_overrides", None) or {}
    if isinstance(ov, dict):
        for k, v in ov.items():
            try:
                i = int(k)
                if 0 <= i < L:
                    ratios[i] = float(v)
            except Exception:
                pass
    return ratios

class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale: float):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float= 1e-5,affine:bool=True,dtype=None,device=None):
        super().__init__()
        self.dim = dim
        self.eps=eps
        self.affine=affine
        self.dtype=dtype
        self.device=device
        if self.affine:
            self.weight=nn.Parameter(torch.ones(dim,dtype=dtype,device=device))
        else:
            self.register_parameter("weight",None)
    def forward(self,x:torch.Tensor):
        x_float=x.float()
        inv_rms=x_float.pow(2).mean(dim=-1,keepdim=True).add(self.eps).rsqrt()
        inv_rms=inv_rms.to(dtype=x.dtype)
        y=inv_rms*x
        if self.affine:
            y=y*self.weight
        return y
class QK_RMSNorm(nn.Module):
    def __init__(self,head_dim:int,num_head:int,eps:float=1e-5,learnable_temp=True):
        super().__init__()
        self.head_dim=head_dim
        self.num_head=num_head
        self.eps=eps

        self.q_norm=RMSNorm(head_dim,eps)
        self.k_norm=RMSNorm(head_dim,eps)

    def forward(self,q:torch.Tensor,k:torch.Tensor):
        q=self.q_norm(q)
        k=self.k_norm(k)

        return q,k

class RoPE(nn.Module):
    def __init__(self,head_dim:int,max_position:int,base:float=2e4,use_NTK:bool=False,train_length:int=2048):
        super().__init__()
        self.head_dim=head_dim
        self.max_position=max_position
        self.base=base
        assert head_dim%2==0,"RoPE head_dim非偶数"
        self.register_buffer("_dev", torch.empty(0), persistent=False)
        index=-torch.arange(0,head_dim,2,dtype=torch.float32,device=self._dev.device)/head_dim
        self.use_NTK=use_NTK
        self.train_length=train_length
        self.last_key=None #only for NTK
        self.last_cos=None
        self.last_sin=None
        self.target_alpha=float(max_position)/float(train_length)
        if use_NTK:
            self.register_buffer("cos_RoPE",None,persistent=False)
            self.register_buffer("sin_RoPE",None,persistent=False)
            self.register_buffer("RoPE_index",index,persistent=False)#shape[d/2]

        else:
            index=base**index #shape[D/2]
            m=torch.arange(max_position,dtype=torch.float32)#shape[max_position]
            freq=torch.einsum("m,i->mi",m,index)#[max_position,D/2]

            tempcos=torch.cos(freq).unsqueeze(0).unsqueeze(0)
            tempsin=torch.sin(freq).unsqueeze(0).unsqueeze(0)
            self.register_buffer("RoPE_index",None,persistent=False)
            self.register_buffer("cos_RoPE",tempcos,persistent=False)
            self.register_buffer("sin_RoPE",tempsin,persistent=False)

    @torch.no_grad()
    def getCosSin(self,start_pos:int,T:int):
        if self.use_NTK:
            key=(start_pos,T,self._dev.device)
            if self.last_key==key and self.last_cos is not None:
                return self.last_cos,self.last_sin
            p_index=torch.arange(start_pos,start_pos+T,dtype=torch.float32,device=self._dev.device)#shape [T]
            len_diff=max(self.max_position-self.train_length,1)
            p=torch.clamp((p_index-self.train_length)/(len_diff),0.0,max=1.0)
            theta_alpha=1+(self.target_alpha-1)*p#shape [T]
            base=self.base*(theta_alpha**(self.head_dim/(self.head_dim-2))) #shape [T]
            index=(base.unsqueeze(1)**self.RoPE_index.unsqueeze(0))
            freq=p_index.unsqueeze(1)*index
            cos=torch.cos(freq).unsqueeze(0).unsqueeze(0)
            sin=torch.sin(freq).unsqueeze(0).unsqueeze(0)
            self.last_sin=sin
            self.last_cos=cos
            self.last_key=key
            return cos,sin
        else:
            cos_temp=self.cos_RoPE[...,start_pos:start_pos+T,:]
            sin_temp=self.sin_RoPE[...,start_pos:start_pos+T,:]
            return cos_temp,sin_temp
    def forward(self,x:torch.Tensor,start_pos:int=0):
        B,H,T,D=x.shape
        x_even=x[...,::2]
        x_odd=x[...,1::2]
        x_RoPE=torch.empty_like(x)
        cos,sin=self.getCosSin(start_pos,T)
        cos=cos.to(x.dtype)
        sin=sin.to(x.dtype)
        x_RoPE[...,::2]=x_even*cos-x_odd*sin
        x_RoPE[...,1::2]=x_even*sin+x_odd*cos
        return x_RoPE



class HF_RoPEBackend(nn.Module):

    def __init__(self, hidden_size:int,num_heads:int,head_dim:int, max_position:int, base:float,
                 rope_type:str, train_length:int,device:torch.device):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.base = float(base)
        self.max_position = int(max_position)
        self.train_length = int(max(1, train_length))

        factor = max(1.0, float(self.max_position) / float(self.train_length))
        cfg = SimpleNamespace()
        cfg.max_position_embeddings = self.max_position
        cfg.rope_theta = self.base
        cfg.head_dim=head_dim
        cfg.rope_scaling = {"rope_type": rope_type, "factor": factor,"original_max_position_embeddings": train_length} if rope_type != "default" else None
        cfg.hidden_size = hidden_size
        cfg.num_attention_heads=num_heads
        rope_config_validation(cfg)
        init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        rope_state = init_fn(cfg,device)
        if isinstance(rope_state, tuple):
            inv_freq, attn_scale = rope_state
        else:
            inv_freq = getattr(rope_state, "inv_freq")
            attn_scale = getattr(rope_state, "attention_factor", 1.0)
        self.register_buffer("inv_freq", inv_freq.to(torch.float32), persistent=False)
        self.attn_scale = float(attn_scale) if attn_scale is not None else None
    @torch.no_grad()
    def build_cos_sin(self, position_ids: torch.Tensor, dtype):
        device = self.inv_freq.device
        pos = position_ids.to(device=device, dtype=torch.float32)      # [B,T]
        inv = self.inv_freq.to(dtype=torch.float32)
        freqs = torch.einsum("bt,d->btd", pos, inv)
        emb = torch.cat([freqs, freqs], dim=-1) # [B,T,d]
        cos = torch.cos(emb).to(dtype)
        sin = torch.sin(emb).to(dtype)
        if self.attn_scale is not None:
            cos = cos * float(self.attn_scale)
            sin = sin * float(self.attn_scale)
        return cos, sin



class TCNBranchPT(nn.Module):
    """
    并联的 TCN 分支（pytorch-tcn），保持 [B,T,H] 形状。
    """
    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 4,
        expansion: float = 0.175,
        kernel_size: int = 9,
        dilations = [1, 2, 4,8],
        dropout: float = 0.05,
        use_norm: str = "weight_norm",
    ):
        super().__init__()

        out_c = max(1, int(round(hidden_size * float(expansion))))
        channels: List[int] = [out_c for _ in range(int(num_blocks))]

        self.tcn = TCN(
            num_inputs=hidden_size,
            num_channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dilation_reset=None,
            dropout=dropout,
            causal=True,
            use_norm=use_norm,
            activation="relu",
            use_skip_connections=False,
            input_shape="NLC",                   # [B,T,H] 输入
            output_projection=hidden_size,
            output_activation=None,
        )
        self.ln_in = nn.LayerNorm(hidden_size)


    @torch.no_grad()
    def new_state(self) -> list[torch.Tensor]:
        """
        返回一个“干净”的初始状态快照（全零/初始 buffer）。
        """
        self.tcn.reset_buffers()
        return [b.clone() if b is not None else None for b in self.tcn.get_buffers()]

    @torch.no_grad()
    def clone_state(self, state: list[torch.Tensor]) -> list[torch.Tensor]:
        return [ (t.clone() if t is not None else None) for t in state ]

    @torch.no_grad()
    def set_state(self, state: list[torch.Tensor]):
        """
        把外部 state 写回到 TCN 模块内部。
        """
        buf = [ (t.clone() if t is not None else None) for t in state ]
        self.tcn.set_buffers(buf)

    @torch.no_grad()
    def get_state(self) -> list[torch.Tensor]:
        """
        读取当前 TCN 模块内部状态（深拷贝）。
        """
        return [b.clone() if b is not None else None for b in self.tcn.get_buffers()]

    @torch.no_grad()
    def reset_state(self):
        self.tcn.reset_buffers()



    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None, inference: bool = False, state_in: list[torch.Tensor] | None = None,return_state: bool = False,):
        # x: [B,T,H], mask: [B,T] (1=valid)

        z = self.ln_in(x)
        if attention_mask is not None:
            z = z * attention_mask.to(z.dtype).unsqueeze(-1)

        if not inference:
            y = self.tcn(z, inference=False)
            if attention_mask is not None:
                y = y * attention_mask.to(y.dtype).unsqueeze(-1)
            if return_state:
                return y, self.get_state()
            return y

        if state_in is not None:
            self.set_state(state_in)
        y = self.tcn(z, inference=True)          # 走库的 streaming 逻辑
        if attention_mask is not None:
            y = y * attention_mask.to(y.dtype).unsqueeze(-1)

        if return_state:
            return y, self.get_state()
        return y
    @staticmethod
    @torch.no_grad()
    def tcn_step_batch(
        tcn_branch: "TCNBranchPT",
        x_btH: torch.Tensor,
        mask_bt: torch.Tensor | None,        # [B,T] or None
        states: list[list[torch.Tensor] | None],  # len=B，每样本一个 state_list
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        B, T, H = x_btH.shape
        ys, new_states = [], []
        for b in range(B):
            x_b = x_btH[b:b+1]
            m_b = None if mask_bt is None else mask_bt[b:b+1]
            st_in = states[b]
            y_b, st_out = tcn_branch(
                x_b, m_b,
                inference=True,
                state_in=st_in,
                return_state=True
            )
            ys.append(y_b)
            new_states.append(st_out)
        y_btH = torch.cat(ys, dim=0)
        return y_btH, new_states
class Mutihead_attention(nn.Module):
    def __init__(self,
                 hidden_size:int,
                 num_heads:int,
                 num_kv_heads:int|None=None,
                 max_position_embeddings:int=16384,
                 RoPE_base:float=2e4,
                 dropout:float=0.0,
                 use_qk_RMSnorm:bool=True,
                 qk_RMSeps:float=1e-5,
                 learnable_temp:bool=True,
                 use_sample_attention:bool=False,
                 rope_type:str="yarn",
                 train_length:int=4096,
                 use_HF_RoPE:bool=True,
                 ):
        super().__init__()
        self.hidden_size=hidden_size
        assert hidden_size%num_heads==0,"hidden size doesn't match n*num_head"
        self.num_heads=num_heads
        if not num_kv_heads:
            self.num_kv_heads=num_heads
        else:
            self.num_kv_heads=num_kv_heads
        self.dropout=dropout
        self.use_qk_RMSnorm=use_qk_RMSnorm
        self.qk_RMSeps=qk_RMSeps
        self.learnable_temp=learnable_temp
        self.dim_perhead=hidden_size//num_heads
        self.w_q=nn.Linear(hidden_size,num_heads*self.dim_perhead,bias=False)
        self.w_k=nn.Linear(hidden_size,self.num_kv_heads*self.dim_perhead,bias=False)
        self.w_v=nn.Linear(hidden_size,self.num_kv_heads*self.dim_perhead,bias=False)
        self.w_o=nn.Linear(num_heads*self.dim_perhead,hidden_size,bias=False)
        if use_HF_RoPE:

            self.register_buffer("_rope_dev_probe", torch.empty(0), persistent=False)
            self.rope_backend = HF_RoPEBackend(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=self.dim_perhead,
                max_position=max_position_embeddings,
                base=RoPE_base,
                rope_type=rope_type,
                train_length=train_length,
                device=self._rope_dev_probe.device
            )
            self.RoPE=None
        else:
            self.RoPE = RoPE(self.dim_perhead, max_position=max_position_embeddings, base=RoPE_base, use_NTK=True,train_length=train_length)
            self.rope_backend = None
        if self.use_qk_RMSnorm:
            self.qk_RMSnorm=QK_RMSNorm(self.dim_perhead,self.num_heads,self.qk_RMSeps)
        else:
            self.qk_RMSnorm=None

        self.use_sample_attention=use_sample_attention
        if self.learnable_temp:
            self.temp_param = nn.Parameter(torch.zeros(num_heads))
            self.temp_range = 0.15
        else:
            self.register_parameter("temp_param", None)
    @staticmethod
    def _reshape(x:torch.Tensor,num_head:int,head_dim:int):
        B,T,_=x.shape
        return x.view(B,T,num_head,head_dim).transpose(1,2).contiguous()
    def sample_attention(self,q,k,v,att_mask=None,is_causal=True,dropout=0.0):
        B,H,T,D=q.shape
        standard=1/math.sqrt(D)
        score=torch.matmul(q,k.transpose(-2,-1))*standard

        if is_causal:
            i=torch.arange(T,device=q.device)
            causal=(i[:,None]>=i[None,:])
            score=score.masked_fill(~causal,float("-inf"))
        if att_mask is not None:
            score=score+att_mask
        score=score-score.max(dim=-1,keepdim=True).values
        att=torch.softmax(score,dim=-1)
        att=F.dropout(att,p=dropout,training=self.training)
        y=torch.matmul(att,v)
        return y

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            is_causal: bool = True,
            past_state: Optional[dict] = None,
            use_cache: bool = False,
            img_len: int | None = None,
    ):
        """
        x: [B,T,C]
        attention_mask: [B,T]，1=有效,0=pad；单样本/无pad时可为None
        past_state: 本层上一次缓存的状态字典，比如
            {
                "attn": {
                    "k": ... [B, Hkv, Lpast, D],
                    "v": ... [B, Hkv, Lpast, D],
                    "len": ... [B],               # 每个样本已经累积的“真实token个数”
                    "valid_mask": ... [B, Lpast]  # True=这个位置是有效token, False=padding
                }
            }
        """

        B, T, C = x.shape

        # 1) 线性投影拿到 q,k,v
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = self._reshape(q, self.num_heads, self.dim_perhead)  # [B, Hq, T, D]
        k = self._reshape(k, self.num_kv_heads, self.dim_perhead)  # [B, Hkv, T, D]
        v = self._reshape(v, self.num_kv_heads, self.dim_perhead)  # [B, Hkv, T, D]

        if self.use_qk_RMSnorm:
            q, k = self.qk_RMSnorm(q, k)

        # 2) 取出过去的 cache
        past_kv = None
        past_len = None  # [B]，每条样本已经累积的真实长度
        past_valid_mask = None  # [B, Lpast]，True表示这个位置是有效token
        if past_state is not None and "attn" in past_state:
            kv = past_state["attn"]
            past_kv = (kv["k"], kv["v"])
            past_len = kv.get("len", None)
            past_valid_mask = kv.get("valid_mask", None)

        # 3) 计算本 step 内的“局部位置”（忽略左填充）
        #    和本 step 的有效位 mask（哪些是真token，而不是pad）
        if attention_mask is not None:
            am_long = attention_mask.to(torch.long)  # [B,T]
            local_pos = (am_long.cumsum(dim=-1) - 1).clamp_min(0)  # [B,T], pad行会复用0
            chunk_valid_mask = am_long.bool()  # [B,T]
        else:
            # 无mask就说明整段都是实 token
            local_pos = torch.arange(T, device=x.device).view(1, T).expand(B, T)
            chunk_valid_mask = torch.ones((B, T), dtype=torch.bool, device=x.device)

        # 4) 计算 RoPE 的全局绝对位置：每个样本自己的offset
        if past_len is not None:
            # past_len: [B]  => [B,1]  broadcast
            offset = past_len.to(device=x.device, dtype=torch.long).view(B, 1)
            pos = local_pos + offset  # [B,T]
        else:
            pos = local_pos  # [B,T]，第一段/prefill

        # 5) 应用 RoPE
        if self.rope_backend is not None:
            cos, sin = self.rope_backend.build_cos_sin(pos, dtype=q.dtype)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        else:
            # 老的自实现 RoPE 只支持统一的 start_pos（标量）。
            # 单batch还好，multi-batch其实不严谨。
            start_pos_scalar = 0
            if past_len is not None:
                # 只能拿第0个样本的offset做近似；多batch会不准，所以尽量别走这个分支。
                start_pos_scalar = int(past_len[0].item())
            q = self.RoPE(q, start_pos=start_pos_scalar)
            k = self.RoPE(k, start_pos=start_pos_scalar)
        if self.learnable_temp and (self.temp_param is not None):

            s = 1.0 + self.temp_range * torch.tanh(self.temp_param).to(q.dtype)
            q = q * s.view(1, self.num_heads, 1, 1)
        # 6) 把新的 k,v 接到过去的 k,v 后面；同时合并 valid_mask
        TotalT = T
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            TotalT = k.size(-2)

        if past_valid_mask is not None:
            full_valid_mask = torch.cat([past_valid_mask, chunk_valid_mask], dim=-1)  # [B, TotalT]
        else:
            full_valid_mask = chunk_valid_mask  # [B,T] in first chunk
        # k_kv/v_kv 用于cache保存（head还是 Hkv）
        k_kv, v_kv = k, v

        # 7) 如果是 GQA，多头扩展到 Hq
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = (
                k.unsqueeze(2)
                .expand(B, self.num_kv_heads, repeat, TotalT, self.dim_perhead)
                .reshape(B, self.num_heads, TotalT, self.dim_perhead)
            )
            v = (
                v.unsqueeze(2)
                .expand(B, self.num_kv_heads, repeat, TotalT, self.dim_perhead)
                .reshape(B, self.num_heads, TotalT, self.dim_perhead)
            )

        # 8) 构造注意力mask，保证：
        #    - 不能看未来（causal）
        #    - 永远不能看 padding（full_valid_mask==False 的地方）
        attn_mask_for_sdpa = None
        use_causal_flag = is_causal
        need_custom_mask = (past_kv is not None) or (attention_mask is not None)
        if need_custom_mask:
            # key_time: 每个 key 位置是“第几个有效 token”（0,1,2,...）
            key_time = full_valid_mask.long().cumsum(dim=-1) - 1          # [B, TotalT]
            big = torch.iinfo(torch.int64).max // 2
            key_time = key_time.masked_fill(~full_valid_mask, big)        # pad 给一个超大时间戳

            # query_time: 当前 chunk 的“逻辑时间”，直接用上面算好的 pos
            # pos: [B, T]，已经是 local_pos (+ past_len)
            query_time = pos.to(dtype=torch.int64)                        # [B, T]

            Bq, Tq = query_time.shape
            assert Bq == B and Tq == T

            qt = query_time.view(B, 1, T, 1)                              # [B,1,T,1]
            kt = key_time.view(B, 1, 1, TotalT)                           # [B,1,1,TotalT]

            # 不能看：
            #   1) padding（full_valid_mask==False）
            #   2) 逻辑时间晚于自己的 token（strict future: key_time > query_time）
            blocked_bool = (kt > qt) | (~full_valid_mask.view(B, 1, 1, TotalT))

            if (img_len is not None) and (img_len > 0) and (past_kv is None) and is_causal:
                q_idx = torch.arange(T, device=x.device)
                k_idx = torch.arange(TotalT, device=x.device)
                q_is_prefix = (q_idx < img_len).view(1, 1, T, 1)
                k_is_prefix = (k_idx < img_len).view(1, 1, 1, TotalT)
                blocked_bool = blocked_bool & ~(q_is_prefix & k_is_prefix)  # prefix<->prefix 双向可见
                blocked_bool = blocked_bool | (q_is_prefix & ~k_is_prefix)  # prefix->text 禁止偷看

            attn_mask_for_sdpa = torch.zeros((B, 1, T, TotalT), dtype=q.dtype, device=x.device)
            attn_mask_for_sdpa = attn_mask_for_sdpa.masked_fill(
                blocked_bool, torch.finfo(q.dtype).min
            )
            use_causal_flag = False

        # 9) 真正做注意力
        if self.use_sample_attention:
            # sample_attention 用的是加法 mask（0 / -inf）
            add_mask = attn_mask_for_sdpa
            y = self.sample_attention(
                q, k, v,
                add_mask,
                use_causal_flag and is_causal,
                self.dropout
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask_for_sdpa,
                dropout_p=(self.dropout if self.training else 0.0),
                is_causal=use_causal_flag,
            )

        # [B,Hq,T,D] -> [B,T,H]
        y = y.transpose(1, 2).contiguous().view(B, T, self.hidden_size)

        # 10) 组装 present cache
        present = None
        if use_cache:
            # 更新每个样本的“真实token计数”
            cur_valid_lengths = chunk_valid_mask.sum(dim=-1).to(dtype=torch.long)  # [B]
            if past_len is not None:
                new_len = past_len + cur_valid_lengths
            else:
                new_len = cur_valid_lengths

            present = {
                "attn": {
                    "k": k_kv,  # [B,Hkv,TotalT,D]
                    "v": v_kv,  # [B,Hkv,TotalT,D]
                    "len": new_len,  # [B]  （真实token累计数）
                    "valid_mask": full_valid_mask,  # [B,TotalT] True=有效token
                }
            }

        return self.w_o(y), present


class Drop_path(nn.Module):
    def __init__(self,drop_p:float=0.0):
        super().__init__()
        self.drop_p=drop_p

    def forward(self,x:torch.Tensor):
        if self.drop_p==0.0 or not self.training:
            return x
        keep=1-self.drop_p
        shape=(x.size(0),)+(1,)*(x.ndim-1)
        mask=torch.rand(shape,device=x.device) <=keep
        return x*(mask.to(dtype=x.dtype))/(keep)

class MLP(nn.Module):
    def __init__(self,hidden_dim:int,dropout,amplify:float=4,use_swiGLU:bool=True):
        super().__init__()
        self.activate=nn.SiLU()
        self.dropout=nn.Dropout(dropout)
        self.use_swiGLU=use_swiGLU
        if use_swiGLU:
            self.amplify=int((amplify*2/3*hidden_dim+63)//64*64)
            self.upsamp=nn.Linear(hidden_dim,self.amplify,bias=False)
            self.swiGate=nn.Linear(hidden_dim,self.amplify, bias=False)
            self.downsamp=nn.Linear(self.amplify,hidden_dim, bias=False)
        else:
            self.amplify=int(amplify*hidden_dim)
            self.upsamp=nn.Linear(hidden_dim,self.amplify, bias=False)
            self.downsamp=nn.Linear(self.amplify,hidden_dim, bias=False)

    def forward(self,x:torch.Tensor):
        if self.use_swiGLU:
            return self.dropout(self.downsamp(self.activate(self.swiGate(x))*self.upsamp(x)))
        return self.dropout(self.downsamp(self.activate(self.upsamp(x))))
class VisionProjector(nn.Module):

    def __init__(
        self,
        vision_dim: int,
        hidden_dim: int,
        use_swiGLU: bool = True,
        dropout: float = 0.0,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        self.use_swiGLU = use_swiGLU
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

        # 1) 1536 -> 1280
        self.proj = nn.Linear(vision_dim, hidden_dim, bias=False)

        if use_rmsnorm:
            # 复用你上面的 RMSNorm
            self.norm = RMSNorm(hidden_dim, eps=1e-5, affine=True)
        else:
            self.norm = nn.LayerNorm(hidden_dim)


    def forward(self, feats: torch.Tensor):
        # feats: [B, Nv, Dv]
        x = self.proj(feats)
        x = self.norm(x)
        return x

class QFormerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        use_rms_norm: bool = False,
    ):
        super().__init__()

        Norm = RMSNorm if use_rms_norm else nn.LayerNorm

        # self-attn over queries
        self.ln_self = Norm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # cross-attn: q = query tokens, kv = vision tokens
        self.ln_cross_q = Norm(hidden_dim)
        self.ln_cross_kv = Norm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # MLP
        self.ln_mlp = Norm(hidden_dim)
        self.mlp = MLP(hidden_dim=hidden_dim, amplify=mlp_ratio, dropout=dropout)

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,               # [B, Nq, H]  learnable queries 或 text-conditioned queries
        kv: torch.Tensor,              # [B, Nv, H]  vision tokens（已经 1536→1280 过了）
        kv_mask: torch.Tensor | None = None,  # [B, Nv]，1=valid, 0=pad
    ) -> torch.Tensor:
        # 1) Self-Attention over q（完全非因果，双向）
        q_norm = self.ln_self(q)
        q_sa, _ = self.self_attn(
            q_norm, q_norm, q_norm,
            need_weights=False,         # 有利于用 fused kernel
        )
        q = q + self.drop(q_sa)

        # 2) Cross-Attention：q ← kv
        q_norm = self.ln_cross_q(q)
        kv_norm = self.ln_cross_kv(kv)

        if kv_mask is not None:
            # key_padding_mask: True = 要 mask 掉
            key_padding_mask = (~kv_mask.bool())   # [B, Nv]
        else:
            key_padding_mask = None

        q_ca, _ = self.cross_attn(
            q_norm, kv_norm, kv_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        q = q + self.drop(q_ca)

        # 3) 小 MLP
        q_mlp = self.mlp(self.ln_mlp(q))
        q = q + q_mlp

        return q    # [B, Nq, H]

class Top1Router(nn.Module):
    def __init__(self,hidden_size:int,num_expert:int,enable_detach:bool=False):
        super().__init__()
        self.decider=nn.Linear(hidden_size,num_expert,bias=True)
        self.num_expert=num_expert
        self.enableDetach = enable_detach
    def forward(self,x:torch.Tensor):

        x=x.detach()
        logit=self.decider(x)
        logit = logit - logit.mean(dim=-1, keepdim=True)
        std = logit.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-3)
        logit=logit / std/1.2
        prob=F.softmax(logit,dim=-1)

        if self.training:
            top_idx=(logit+0.4*gumbel_noise(logit)).topk(2, dim=-1).indices
        else:
            top_idx = prob.topk(2, dim=-1).indices
        top_p = prob.gather(1, top_idx)
        return top_idx,top_p,prob

class MoEMLP(nn.Module):
    def __init__(self,hidden_dim:int,dropout,num_expert:int,amplify:float=4,use_swiGLU:bool=True,cap_factor:float=1.25,enable_detach:bool=False,n_shared:int=1):
        super().__init__()
        self.num_expert=num_expert
        self.expert=nn.ModuleList([MLP(hidden_dim,dropout,amplify,use_swiGLU) for i in range(num_expert)])

        self.router=Top1Router(hidden_dim,num_expert,enable_detach)
        self.cap_factor=cap_factor
        self.enableDetach=enable_detach
        self.register_buffer("top2_alpha", torch.tensor(0.0))

        self.n_shared = n_shared
        self.a, self.b = 0.6, 1.4
        if n_shared>0:
            self.shared = nn.ModuleList([MLP(hidden_dim, dropout, amplify, use_swiGLU) for _ in range(n_shared)])


            self.shared_gate = nn.Linear(hidden_dim, 1)

            nn.init.zeros_(self.shared_gate.weight)
            nn.init.zeros_(self.shared_gate.bias)
        else:
            self.shared = None
            self.shared_gamma = None
    def forward(self,x:torch.Tensor):

        B,T,H=x.shape
        N=B*T
        x=x.view(N,H)

        device=x.device
        e,prob_selected,prob=self.router(x)
        e1, e2 = e[:,0], e[:,1]
        w1, w2 = prob_selected[:,0:1], prob_selected[:,1:2]

        e1_orderd,e1_idx=e1.sort()
        sorted1_x=x[e1_idx]
        sorted1_p=w1[e1_idx]
        count1=torch.bincount(e1_orderd,minlength=self.num_expert) # [E]

        if self.training:
            capacity=max(1,math.ceil(self.cap_factor*N/self.num_expert))
        else:
            capacity=int(1e12)
        keep_per_exp1 = torch.minimum(count1, torch.full_like(count1, capacity))
        prefix=torch.cumsum(count1,dim=0) # [E]
        start=prefix-count1
        end=prefix
        realend=start+keep_per_exp1
        prob_32=prob.to(torch.float32)
        imp=prob_32.sum(dim=0)
        imp_hat=imp/imp.sum().clamp_min(1e-6)
        load=keep_per_exp1.to(torch.float32)
        load_hat=load/load.sum().clamp_min(1e-6)
        log_imp=(imp_hat.clamp_min(1e-9)).log()
        log_load=(load_hat.clamp_min(1e-9)).log()
        log_u=-math.log(self.num_expert)
        L_imp=(imp_hat*(log_imp-log_u)).sum()
        L_load=(load_hat*(log_load-log_u)).sum()
        aux_loss=L_imp+L_load

        aux_loss= aux_loss/ (2.0 * (-log_u) + 1e-12)
        sorted1_y=torch.zeros_like(sorted1_x)

        for j in range(self.num_expert):
            sj=start[j]
            rj=realend[j]
            ej=end[j]
            if ej==sj:
                continue
            if rj>sj:
                x_kept=sorted1_x[sj:rj]
                y_kept=self.expert[j](x_kept)
                p_kept=sorted1_p[sj:rj]
                if self.enableDetach:
                    p_kept=p_kept.detach()
                else:
                    # 只放行 router_grad_frac 的梯度（直通估计 + 缩放）
                    p_kept = p_kept.detach() + GradScale.apply(p_kept - p_kept.detach(), 0.1)
                y_kept=y_kept*p_kept
                sorted1_y[sj:rj]=y_kept
            if ej>rj:
                pass

        inv1_idx=torch.argsort(e1_idx)
        y=sorted1_y[inv1_idx].reshape(B,T,H)


        if self.shared is not None:
            shared_y = 0
            x_btH = x.reshape(B, T, H)
            for se in self.shared:
                shared_y = shared_y + se(x_btH)
            shared_y = shared_y / self.n_shared

            gamma_tok = torch.sigmoid(self.shared_gate(x_btH))
            gamma_tok = self.a + (self.b - self.a) * gamma_tok
            y = y + gamma_tok * shared_y
        if self.training:
            return y,aux_loss
        return y
class LoraAdapter(nn.Module):
    def __init__(self,input_dim:int,out_dim:int,rank:int=8,dropout:float=0.1,alpha:float=16):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=out_dim
        self.r=rank
        self.alpha=alpha
        self.amplify=alpha/rank
        self.A=nn.Linear(input_dim,rank,bias=False)
        self.B=nn.Linear(rank,out_dim,bias=False)
        self.dropout=nn.Dropout(dropout)
        nn.init.zeros_(self.B.weight)
    def forward(self,x:torch.Tensor):
        x=self.A(x)
        x=self.dropout(x)
        x=self.B(x)*self.amplify
        return x

class LoraLinear(nn.Module):
    def __init__(self,base_linear:nn.Linear,mode:Literal["exclusive", "additive", "weighted"]="exclusive",cap_norm:float|None = None,global_scale:float=1.0):
        super().__init__()
        assert isinstance(base_linear,nn.Linear),"LoraLinear type wrong"
        self.base=base_linear
        self.input_dim=base_linear.in_features
        self.out_dim=base_linear.out_features
        self.adapters=nn.ModuleDict()
        self.active=[]
        self.mode=mode
        self.weights={}
        self.cap_norm=cap_norm
        self.global_scale=global_scale
        self.dtype=base_linear.weight.dtype
        self.device=base_linear.weight.device
    def set_mode(self,mode:str):
        self.mode=mode
    def set_cap_norm(self,cap):
        self.cap_norm=cap
    def set_global_scale(self,s):
        self.global_scale=s
    def register_adapter(self,name:str,rank:int=8,dropout:float=0.1,alpha:float=16.0,state_dict:dict|None=None,weight:float=1.0):
        if name in self.adapters:
            raise ValueError(f"LoRA adapter {name} already registered")
        adapter=LoraAdapter(self.input_dim,self.out_dim,rank,dropout,alpha).to(self.dtype).to(self.device)
        self.adapters[name]=adapter
        if state_dict is not None:
            miss,unexpect=adapter.load_state_dict(state_dict,strict=False)
            if miss:
                print(f"Lora {name}: missing key {miss} ")
            if unexpect:
                print(f"Lora {name}: unexpect key {unexpect} ")
        if name not in self.weights :
            self.weights[name]=weight
    def activate(self,name:str,exclusive:bool=False):
        if name not in self.adapters:
            raise KeyError(f"LoRA adapter {name} not registered.")
        if exclusive or self.mode=="exclusive":
            self.active=[name]
        else:
            self.active.append(name)
    def deactivate(self,name:str):
        if name not in self.adapters:
            raise KeyError(f"LoRA adapter {name} not registered.")
        self.active=[n for n in self.active if n!=name]
    def unload(self,name:str):
        self.deactivate(name)
        if name in self.weights:
            del self.weights[name]
        if name in self.adapters:
            del self.adapters[name]
    def set_active(self,names:list[str]):
        for name in names:
            if name not in self.adapters:
                raise KeyError(f"LoRA adapter {name} not registered.")
        self.active=names
    def set_weight(self,name:str,w:float):
        if name not in self.adapters:
            raise KeyError(f"LoRA adapter {name} not registered.")
        self.weights[name]=w
    def forward(self,x:torch.Tensor):
        y=self.base(x)
        if not self.active:
            return y
        z = None
        if self.mode=="additive":
            for n in self.active:
                adp=self.adapters[n]
                d=adp(x)
                z = d if z is None else (z + d)
        if self.mode=="exclusive":
            z=self.adapters[self.active[0]](x)
        if self.mode=="weighted":
            for n in self.active:
                adp=self.adapters[n]
                w=self.weights.get(n,1.0)
                if w==0.0:
                    continue
                d=adp(x)*w
                z = d if z is None else (z + d)
        if z is None:
            return y
        if self.cap_norm is not None:
            norm=torch.linalg.vector_norm(z,dim=-1,keepdim=True)
            scale=torch.clamp(self.cap_norm/(norm+1e-6),max=1.0)
            z=z*scale
        if self.global_scale is not None:
            z=z*self.global_scale

        return y+z




class Transformer_block(nn.Module):
    def __init__(self,
                 hidden_size:int=640,
                 num_heads:int=10,
                 rms_eps: float=1e-5,
                 dropout:float=0.1,
                 num_kv_heads:int|None=None,
                 use_qk_RMSnorm: bool = True,
                 learnable_temp: bool = True,
                 use_affine:bool=True,
                 qkrms_eps:float|None=None,
                 use_sampleatt:bool=False,
                 rope_type:str="yarn",
                 training_length:int=4096,
                 use_swiGLU: bool = True,
                 mlp_ratio: float = 4.0,
                 RoPE_base:float=2e4,
                 max_position_embeddings:int=16384,
                 drop_path:float=0.1,
                 resid_dropout:float=0.05,
                 use_moe_layer: bool=False,
                 moe_use_detach:bool=False,
                 moe_cap_factor:float=1.25,
                 moe_num_expert:int=4,
                 use_ssm:bool=False,
                 mamba_d_state: int = 64,
                 mamba_d_conv: int = 4,
                 mamba_expand = 1.5,
                 use_HF_ROPE=True,
                 use_tcn:bool=False
                 ):
        super().__init__()
        self.register_buffer("_env",torch.empty(0))
        self.gamma_att = nn.Parameter(torch.ones(1) * 1.0)
        self.gamma_mlp = nn.Parameter(torch.ones(1) * 1.0)
        if qkrms_eps is None:
            self.qkrms_eps=rms_eps
        else:
            self.qkrms_eps = qkrms_eps
        if num_kv_heads is None:
            self.num_kv_heads=num_heads
        else:
            self.num_kv_heads= num_kv_heads
        self.attRMS_norm=RMSNorm(hidden_size,rms_eps,use_affine,self._env.dtype,self._env.device)
        self.use_ssm=bool(use_ssm and HAS_MAMBA2)
        if self.use_ssm:
            cfg2 = Mamba2Config(
                hidden_size=hidden_size,  # 1280
                num_heads=10,
                head_dim=192,  # 256
                n_groups=5,  # 5
                state_size= mamba_d_state,  # 64
                conv_kernel=mamba_d_conv,  # 4
                expand=mamba_expand,  # 1.5
                num_hidden_layers=1,
                use_cache=True,
                use_bias=False,
                norm_before_gate=True,
                rms_norm=True,
                vocab_size=1,
            )
            self.ssm = Mamba2Model(cfg2)

        else:
            self.selfatt=Mutihead_attention(hidden_size,num_heads,self.num_kv_heads,max_position_embeddings,RoPE_base,dropout,use_qk_RMSnorm,self.qkrms_eps,learnable_temp,use_sampleatt,rope_type,training_length,use_HF_ROPE)


        self.is_moe = use_moe_layer

        self.gamma_min = 0.6
        self.gamma_max = 1.2
        if self.is_moe:
            self.MLP = MoEMLP(hidden_size,dropout,moe_num_expert,mlp_ratio,use_swiGLU,moe_cap_factor,moe_use_detach)
        else:
            self.MLP = MLP(hidden_size, dropout, mlp_ratio, use_swiGLU)
        self.droppath=Drop_path(drop_path)
        self.MLPrms_norm=RMSNorm(hidden_size,rms_eps,use_affine,self._env.dtype,self._env.device)
        self.residual_dropout=nn.Dropout(resid_dropout)
        self.use_tcn = use_tcn
        if self.use_tcn:
            self.tcnRMS_norm = RMSNorm(hidden_size, rms_eps, use_affine, self._env.dtype, self._env.device)
            self.tcn_branch = TCNBranchPT(
                hidden_size=hidden_size)
            self.tcc_gate = nn.Parameter(torch.tensor(-2.1972246, dtype=self._env.dtype))

    def _bounded(self, raw):
        # γ = γ_min + (γ_max-γ_min) * sigmoid(raw)
        return self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(raw)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            is_causal: bool = True,
            past_state: dict | None = None,
            use_cache: bool = False,
            img_len: int | None = None,
    ):
        # x: [B, T, H]
        B, T, H = x.shape
        past_state = past_state or {}


        present_state = {} if use_cache else None

        # =====================
        # 1. Attention or SSM
        # =====================
        if not self.use_ssm:

            attout, att_present = self.selfatt(
                self.attRMS_norm(x),
                attention_mask,
                is_causal,
                past_state=past_state,
                use_cache=use_cache,
                img_len=img_len,
            )
            if use_cache and att_present is not None:
                present_state.update(att_present)
        else:
            # --------- SSM / Mamba2 分支 ---------


            if (attention_mask is not None) and (attention_mask.device != x.device):
                attention_mask = attention_mask.to(x.device)

            # 取上一次保存的状态
            ssm_fields = past_state.get("ssm", None) if past_state is not None else None
            if ssm_fields is not None:
                ssm_cache_prev = ssm_fields.get("cache_params", None)
                cache_pos_prev = ssm_fields.get("cache_position", None)  # shape [B] or None
            else:
                ssm_cache_prev = None
                cache_pos_prev = None

            # 对输入做和训练一致的norm/layout
            hs = self.attRMS_norm(x)
            # 你训练时为了layout对齐做的 transpose-contiguous-transpose，就保留
            hs = hs.transpose(1, 2).contiguous().transpose(1, 2).contiguous()  # [B,T,H] 但内存布局是你训练期喜欢的

            fastpath_even_in_eval = (not use_cache)

            if fastpath_even_in_eval:
                was_mode = self.ssm.training
                try:
                    # 临时切到 train()：只影响 ssm 子模块，不改变梯度开关（由 no_grad 决定）
                    self.ssm.train(True)
                    out = self.ssm(
                        inputs_embeds=hs,
                        attention_mask=attention_mask,
                        cache_params=None,
                        cache_position=None,
                        use_cache=False,
                    )
                finally:
                    # 恢复到进入前的模式（非常重要）
                    self.ssm.train(was_mode)
            else:
                # 增量解码：保持外部模式（通常 eval）
                out = self.ssm(
                    inputs_embeds=hs,
                    attention_mask=attention_mask,
                    cache_params=ssm_cache_prev,
                    cache_position=cache_pos_prev,
                    use_cache=True,
                )

            # SSM 输出
            attout = out.last_hidden_state.to(x.dtype)  # [B,T,H] 这就是这一层SSM分支的本轮输出
            ssm_noise=float(0.04)
            if self.training:

                s = torch.empty((), device=attout.device, dtype=attout.dtype).uniform_(1.0 - ssm_noise, 1.0)
                c = 1.0 / (1.0 - 0.5 * ssm_noise)
                attout = attout * (s * c)
            # 我们要把“下一轮要用的 cache_state”记下来，连同正确的下一拍位置
            if use_cache:
                # 计算下一轮的 cache_position 向量
                #
                # case 1: 这是 prefill（整段prompt喂进去）
                #   - 此时 cache_pos_prev 是 None
                #   - 我们需要根据这一轮喂了多少真实token(不算左pad)来初始化它
                # case 2: 这是增量 streaming（一般 T=1）
                #   - 我们已有 cache_pos_prev 了
                #   - 下一拍位置 = 旧位置 + 当前这次我们真正追加的token数
                #
                if cache_pos_prev is None:
                    if attention_mask is not None:
                        # attention_mask: [B,T]，是1的地方才是真正喂给模型的有效token
                        step_len_vec = attention_mask.to(torch.int64).sum(dim=-1)  # [B], 每条样本本轮喂了多少有效token
                    else:
                        # 没有mask就说明没有pad，整段都有效
                        step_len_vec = torch.full(
                            (B,),
                            T,
                            device=x.device,
                            dtype=torch.long,
                        )
                    # prefill之后的"下一拍位置"就是各自的有效长度
                    next_cache_pos = step_len_vec  # shape [B]
                else:
                    # streaming 步：通常 T==1
                    step_len_vec = torch.full(
                        (B,),
                        T,
                        device=x.device,
                        dtype=torch.long,
                    )
                    # 注意这里是 elementwise 加法，不是广播成同一个数字
                    # cache_pos_prev.shape == [B]
                    next_cache_pos = cache_pos_prev + step_len_vec  # [B]

                present_state["ssm"] = {
                    "cache_params": out.cache_params,      # HF给我们的新状态
                    "cache_position": next_cache_pos,      # 我们手工维护的 per-sample 时钟
                }

        # =====================
        # 2. 可选 TCN 分支
        # =====================
        if self.use_tcn:
            x_norm = self.tcnRMS_norm(x)

            if use_cache:
                tcn_states_in = past_state.get("tcn", None)
                if tcn_states_in is None:
                    tcn_states_in = [None] * B

                if B == 1:
                    state_in = tcn_states_in[0] if tcn_states_in is not None else None
                    tcn_y, tcn_state_out = self.tcn_branch(
                        x_norm,
                        attention_mask,
                        inference=True,
                        state_in=state_in,
                        return_state=True,
                    )
                    present_state["tcn"] = [tcn_state_out]
                else:
                    tcn_y, tcn_states_out = TCNBranchPT.tcn_step_batch(
                        self.tcn_branch,
                        x_norm,
                        attention_mask,
                        tcn_states_in,
                    )
                    present_state["tcn"] = tcn_states_out
            else:
                tcn_y = self.tcn_branch(
                    x_norm,
                    attention_mask,
                    inference=False,
                )

            tcn_y = tcn_y.to(x.dtype)


            gate = 0.2 + (0.7 - 0.2) * torch.sigmoid(self.tcc_gate)
            attout = (1.0 - gate) * attout + gate * tcn_y

        # =====================
        # 3. 残差 + MLP
        # =====================
        attout = attout * self._bounded(self.gamma_att)
        x = x + self.droppath(self.residual_dropout(attout))

        if self.is_moe and self.training:
            mlp_out, aux_loss = self.MLP(self.MLPrms_norm(x))
            mlp_out = mlp_out * self._bounded(self.gamma_mlp)
        else:
            normed = self.MLPrms_norm(x)
            mlp_out = self.MLP(normed)
            mlp_out = mlp_out * self._bounded(self.gamma_mlp)
            aux_loss = x.new_zeros(())

        x = x + self.droppath(self.residual_dropout(mlp_out))

        return x, aux_loss, present_state


class TinyLLM(nn.Module, GenerationMixin):
    main_input_name = "input_ids"
    _is_stateful = True

    def __init__(self, cfg: Config):
        super().__init__()
        self._train_step = 0
        self.cfg = cfg
        self.config = cfg
        self.generation_config = GenerationConfig(
            eos_token_id=cfg.eos_token_id,
            pad_token_id=cfg.pad_token_id,
        )

        self.end_text_tok_id = cfg.eos_token_id
        self.eso_loss_radio = cfg.eso_loss_radio
        self.drop_high_loss = cfg.drop_high_loss
        self.adaptive_cutoffs = getattr(self.cfg, "adaptive_cutoffs", [20000, 60000])
        self.adaptive_div = getattr(self.cfg, "adaptive_div", 4.0)
        self.adaptive_calibrate_every = 4
        hidden_size = cfg.hidden_size
        vocab_size = cfg.vocab_size
        self.tok_embed = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.02)
        self.embdrop = nn.Dropout(cfg.embeddingdropout)
        droppath_list = [(cfg.drop_path * i / max(cfg.num_hidden_layers - 1, 1)) for i in
                         range(cfg.num_hidden_layers)]
        mlp_ratio_by_layer = _build_mlp_ratio_schedule(self.cfg)
        self.blocks = nn.ModuleList([Transformer_block(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            rms_eps=cfg.rms_norm_eps,
            dropout=cfg.dropout,
            num_kv_heads=cfg.num_key_value_heads,
            use_qk_RMSnorm=cfg.use_qk_norm,
            learnable_temp=cfg.learnable_temp,
            use_affine=cfg.use_affine,
            qkrms_eps=cfg.qkrms_norm_eps,
            use_sampleatt=cfg.use_sampleatt,
            rope_type=cfg.rope_type,
            training_length=cfg.train_maxlength,
            use_swiGLU=cfg.use_swiGLU,
            mlp_ratio=mlp_ratio_by_layer[i],
            RoPE_base=cfg.RoPE_base,
            max_position_embeddings=cfg.max_position_embeddings,
            drop_path=droppath_list[i],
            use_tcn=(cfg.use_tcn and (i in cfg.tcn_layers)),
            use_ssm=(cfg.use_ssm and (i in cfg.ssm_layers)),
            resid_dropout=cfg.residual_dropout,
            use_moe_layer=(getattr(cfg, "use_moe", False) and (i in cfg.moe_layers)),
            moe_use_detach=cfg.moe_use_detach,
            moe_cap_factor=cfg.moe_cap_factor,
            moe_num_expert=cfg.num_expert,
            use_HF_ROPE=True
        ) for i in range(cfg.num_hidden_layers)])
        self.register_buffer("_env0", torch.empty(0))
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.use_affine, self._env0.dtype,
                                  self._env0.device)
        self.lm_bias = nn.Parameter(torch.zeros(vocab_size))
        self._kd_seen = 0
        self._kd_hit = 0
        self.use_vision = bool(getattr(cfg, "use_vision", False))
        self.max_vision_tokens = int(getattr(cfg, "max_vision_tokens", 0))
        self.vision_feature_dim = getattr(cfg, "vision_feature_dim", None)
        self.num_query_tokens = cfg.num_query_tokens
        if self.use_vision:
            assert self.vision_feature_dim is not None, \
                "cfg.vision_feature_dim must be set when cfg.use_vision=True"

            # ---- 1) 轻量 projector: vision_feature_dim -> hidden_size ----
            self.vision_projector = VisionProjector(
                vision_dim=self.vision_feature_dim,
                hidden_dim=cfg.hidden_size,
                dropout=getattr(cfg, "vision_dropout", 0.0),
                use_rmsnorm=getattr(cfg, "vision_use_rmsnorm", True),
            )

            # 可选：给整张图一个 type embedding（加到 patch / q 上都行）
            self.vision_type_embed = nn.Parameter(torch.zeros(cfg.hidden_size))

            # ---- 2) Q-Former 配置（默认四层 encoder-style block） ----
            self.qformer_num_layers = int(getattr(cfg, "qformer_num_layers", 4))
            self.qformer_mlp_ratio = float(getattr(cfg, "qformer_mlp_ratio", 2.0))
            self.qformer_dropout = float(getattr(cfg, "vision_dropout", 0.0))
            self.qformer_use_rmsnorm = bool(getattr(cfg, "vision_use_rmsnorm", True))

            # ---- 3) learnable queries（工业界标配）----
            # [1, Nq, H]，每张图复制一份；初始化用 1/sqrt(H) 缩一下
            self.query_embed = nn.Parameter(
                torch.randn(1, self.num_query_tokens, cfg.hidden_size) / math.sqrt(cfg.hidden_size)
            )

            # Q-Former Block: Self-Attn + Cross-Attn + MLP（全是 encoder-style）
            self.qformer_blocks = nn.ModuleList(
                [
                    QFormerBlock(
                        hidden_dim=cfg.hidden_size,
                        num_heads=cfg.num_attention_heads,
                        mlp_ratio=self.qformer_mlp_ratio,
                        dropout=self.qformer_dropout,
                        use_rms_norm=self.qformer_use_rmsnorm,
                    )
                    for _ in range(self.qformer_num_layers)
                ]
            )

            # 顶层再来一个 final norm（跟主干一样风格）
            self.qformer_final_norm = RMSNorm(
                cfg.hidden_size, cfg.rms_norm_eps, cfg.use_affine
            )
            self.num_vision_views = int(getattr(cfg, "num_vision_views", 5))
            self.num_vision_view_types=2
            # 你现在 global_pos 实际更像“thumb 网格上的 index”（0..Nv-1）。
            # 为了兼容：直接给到 1024（32*32），就算你 unshuffle 后 Nv=256 也没问题（pos 仍 <1024）。
            self.vision_grid_tokens = int(getattr(cfg, "vision_grid_tokens", 1024))

            # (a) view embedding：5 张图 = 5 个 learnable embedding（你要的那个）
            self.vision_view_embed = nn.Embedding(self.num_vision_view_types, cfg.hidden_size)

            # (b) global position embedding：离散 gpos -> embedding
            self.vision_pos_embed = nn.Embedding(self.vision_grid_tokens, cfg.hidden_size)
            nn.init.normal_(self.vision_view_embed.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.vision_pos_embed.weight, mean=0.0, std=0.02)

            # (c) FiLM：把连续的 (x_norm, y_norm, dx, dy) -> (gamma, beta)
            # 工业里常见：小 MLP 输出 2H，然后做 proj_vis = proj_vis*(1+γ)+β
            film_hidden = int(getattr(cfg, "vision_film_hidden", max(32, cfg.hidden_size // 4)))
            self.vision_film_mlp = nn.Sequential(
                nn.Linear(4, film_hidden, bias=True),
                nn.SiLU(),
                nn.Linear(film_hidden, 2 * cfg.hidden_size, bias=True),
            )
            # 关键：最后一层置零，保证初始化“完全不影响”
            nn.init.normal_(self.vision_film_mlp[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.vision_film_mlp[-1].bias)

            # (d) 三个 gate：你担心 gate 歪掉回不来 → 用 tanh gate + 小 scale
            # tanh(0)=0：初始化就是“零注入”，非常稳
            self.vision_gate_view = nn.Parameter(torch.tensor(0.2))
            self.vision_gate_pos = nn.Parameter(torch.tensor(0.2))
            self.vision_gate_film = nn.Parameter(torch.tensor(0.2))

            # FiLM 输出再乘一个小尺度，防止极端（比 hard-clip 更工业）
            self.vision_film_scale = float(getattr(cfg, "vision_film_scale", 0.10))


        else:
            self.vision_projector = None
            self.vision_type_embed = None
            self.query_embed = None
            self.qformer_blocks = None
            self.qformer_final_norm = None

    def encode_image_with_qformer(
            self,
            vision_feats: torch.Tensor,  # [B,5,1024,Dv] or [B,1024,Dv]
            vision_mask: torch.Tensor ,  # [B,5,1024]    or [B,1024]
            global_pos: torch.Tensor,  # [B,5,1024]    or [B,1024]
            global_off: torch.Tensor,  # [B,5,1024,4]  or [B,1024,4]
    ) -> torch.Tensor:
        """
        只跑一次 QFormer：
          - 把 5 个 view 沿 token 维拼成 5120（可选 + 4 sep）
          - 返回 q_img: [B, Nq, H]
        """

        def gate(p: torch.Tensor) -> torch.Tensor:
            return torch.tanh(p)

        if vision_mask is None or global_pos is None or global_off is None:
            raise ValueError("vision_mask/global_pos/global_off must be provided when vision_feats is provided")

        # ---- 0) 统一维度：老数据 [B,Nv,Dv] -> [B,1,Nv,Dv] ----
        if vision_feats.ndim == 3:
            vision_feats = vision_feats.unsqueeze(1)
            if vision_mask is not None:
                vision_mask = vision_mask.unsqueeze(1)

            if global_pos is not None:
                global_pos = global_pos.unsqueeze(1)
            if global_off is not None:
                global_off = global_off.unsqueeze(1)

        B, V, Nv, Dv = vision_feats.shape

        if V != self.num_vision_views:
            # 你要的就是 5；不想 silent 就直接报
            raise ValueError(f"Expected V={self.num_vision_views} views, got V={V}")

        vision_mask = vision_mask.to(torch.long)
        # ---- 1) 你不想截断：那就“若 max_vision_tokens 不够就报错” ----

        total_kv = V * Nv
        if self.max_vision_tokens and total_kv > self.max_vision_tokens:
            raise ValueError(f"vision tokens {total_kv} exceed max_vision_tokens={self.max_vision_tokens}. "
                             f"Increase max_vision_tokens.")

        # ---- 2) 构造每个 token 的 view_id: [B, V*Nv] ----
        view_ids = torch.zeros((B, V, Nv), device=vision_feats.device, dtype=torch.long)
        view_ids[:, -1, :] = 1  # thumb=1（最后一张）
        view_ids = view_ids.reshape(B, V * Nv)

        # ---- 3) 先把 5 段拼起来（先拼 raw Dv / mask / meta）----
        kv_raw = vision_feats.reshape(B, V * Nv, Dv)
        kv_mask = vision_mask.reshape(B, V * Nv).to(torch.long)
        gp = global_pos.reshape(B, V * Nv).to(torch.long)
        go = global_off.reshape(B, V * Nv, -1).to(torch.float32)

        # ---- 4) projector：Dv -> H（复用你已有 projector 接口）----
        proj_vis = self.vision_projector(kv_raw)  # [B, V*Nv, H]
        if self.vision_type_embed is not None:
            proj_vis = proj_vis + self.vision_type_embed.view(1, 1, -1)

        H = proj_vis.size(-1)
        dtype = proj_vis.dtype

        # ---- 6) 加 view embedding（工业标准：加法 + gate）----
        g_view = gate(self.vision_gate_view).to(dtype)
        proj_vis = proj_vis + g_view * self.vision_view_embed(view_ids).to(dtype)

        # ---- 7) 加 global_pos embedding（离散 index）----
        g_pos = gate(self.vision_gate_pos).to(dtype)
        if gp.min() < 0 or gp.max() >= self.vision_pos_embed.num_embeddings:
            raise ValueError(f"global_pos out of range: [{gp.min().item()}, {gp.max().item()}], "
                             f"num_embeddings={self.vision_pos_embed.num_embeddings}")
        proj_vis = proj_vis + g_pos * self.vision_pos_embed(gp).to(dtype)

        # ---- 8) FiLM：连续 (x_norm,y_norm,dx,dy) 调制 ----
        g_film = gate(self.vision_gate_film).to(dtype)
        s = (self.vision_film_scale * g_film).to(dtype)

        go_t = torch.tanh(go)  # [-1,1]，x_norm/y_norm 本来就在 [-1,1]，tanh 基本不影响
        film = self.vision_film_mlp(go_t).to(dtype)  # [B,N,2H]
        gamma, beta = film.chunk(2, dim=-1)
        gamma = torch.tanh(gamma) * s
        beta = torch.tanh(beta) * s
        proj_vis = proj_vis * (1.0 + gamma) + beta

        # ---- 9) mask：padding token 清零（你原来也是这么干的思路）----
        proj_vis = proj_vis * kv_mask.to(dtype).unsqueeze(-1)

        # ---- 10) QFormer：只跑一次 ----
        q = self.query_embed.expand(B, -1, -1)  # [B,Nq,H]
        for blk in self.qformer_blocks:
            q = blk(q, proj_vis, kv_mask=kv_mask)
        q = self.qformer_final_norm(q)  # [B,Nq,H]

        # 如果整条 kv 全是 0，别污染：q 直接清零
        has_any = (kv_mask.sum(dim=1, keepdim=True) > 0).to(dtype)  # [B,1]
        q = q * has_any.view(B, 1, 1)

        return q

    @property
    def device(self):

        return next(self.parameters()).device

    def _project_subset_logits(self, x, idx_btK):
        """
        x:        [B, T, H]  final hidden
        idx_btK:  [B, T, K]  teacher top-k indices (Long)
        return:   [B, T, K]  student logits on these K words
        """
        B, T, H = x.shape
        K = idx_btK.size(-1)
        W = self.tok_embed.weight.float()  # [V, H]（绑权重）
        # 展平 + unique 只算一次
        idx_flat = idx_btK.to(device=W.device, dtype=torch.long).reshape(-1)
        uniq, inv = torch.unique(idx_flat, sorted=False, return_inverse=True)

        b = self.lm_bias.float()  # [V]

        W_sub = W[uniq]  # [U, H]
        b_sub = b[uniq] if b is not None else None

        x_flat = x.reshape(B * T, H).float()  # [B*T, H]
        logits_sub = x_flat @ W_sub.t()  # [B*T, U]
        if b_sub is not None:
            logits_sub = logits_sub + b_sub  # broadcast

        # 把 U 映回 B*T*K 的排列
        logits_btK = logits_sub.gather(1, inv.view(B * T, K)).view(B, T, K)
        return logits_btK

    def _mask_shortlist_dups(self, short_idx, logits_short):
        """
        short_idx:   [B,T,Kp]  token id 列表
        logits_short:[B,T,Kp]  对应 logits
        返回：logits_short'，把重复 token（除首个）置为 -inf
        """
        B, T, Kp = short_idx.shape
        # 展平成 [BT, Kp]
        si = short_idx.view(B * T, Kp)
        lg = logits_short.view(B * T, Kp)

        # 对每行排序以便找重复
        vals, order = torch.sort(si, dim=-1)  # [BT,Kp]
        dup = torch.zeros_like(vals, dtype=torch.bool)  # [BT,Kp]
        dup[:, 1:] = (vals[:, 1:] == vals[:, :-1])  # 重复（与左邻相等）标 True

        # 把“排序后重复位置”映回“原列顺序”
        inv = torch.empty_like(order)
        inv.scatter_(1, order, torch.arange(Kp, device=order.device).unsqueeze(0).expand_as(order))
        dup_in_orig = dup.gather(1, inv)  # [BT,Kp] True 表示该列是重复的第二个或以后
        dup_in_orig[:, 0] = False
        # 屏蔽重复列
        lg = lg.masked_fill(dup_in_orig, float('-inf'))

        return lg.view(B, T, Kp)

    @torch.no_grad()
    def _debug_kd_batch(self, kd_idx, kd_val, kd_mask, labels, tau=1.5, K_print=5, max_print=10):
        """
        简要体检：命中率、label 的 rank 分布、概率归一偏差、抽样打印
        kd_idx: [B,T,K] long
        kd_val: [B,T,K] float
        kd_mask: [B,T]  bool/byte
        labels: [B,T]   long
        """
        device = kd_val.device
        B, T, K = kd_idx.shape
        m = kd_mask.bool()
        n_valid = m.sum().item()
        if n_valid == 0:
            print("[KD] no valid positions in mask.")
            return

        # 1) teacher 概率（温度）
        t_prob = torch.softmax(kd_val[m] / tau, dim=-1)  # [M,K]
        prob_sum = t_prob.sum(dim=-1)  # should ≈1
        sum_err_mean = (prob_sum - 1.0).abs().mean().item()

        # 2) 是否排序（如果没排序取 argmax 定义 top1）
        # 这里的 top1 是 kd_val 最大对应的 idx
        argmax_in_k = kd_val[m].argmax(dim=-1)  # [M]
        top1_idx = kd_idx[m].gather(1, argmax_in_k.view(-1, 1)).squeeze(1)  # [M]

        # 3) 命中率（label 是否在 top-k 内）
        lab = labels[m]
        hits_any = (kd_idx[m] == lab.view(-1, 1)).any(dim=-1)  # [M]
        hit_rate = hits_any.float().mean().item()

        # 4) label 的 rank（若在 top-k 内）
        # rank 定义：按 kd_val 降序后的名次；若无序，先对 kd_val 排序
        # 索引到排序后的顺序
        sorted_vals, order = torch.sort(kd_val[m], dim=-1, descending=True)  # [M,K]
        sorted_idx = kd_idx[m].gather(1, order)  # [M,K]
        # 找出 label 在 sorted_idx 的位置
        eq = (sorted_idx == lab.view(-1, 1))  # [M,K]
        has_rank = eq.any(dim=-1)  # [M]
        ranks = torch.where(has_rank, eq.float().argmax(dim=-1), torch.full_like(argmax_in_k, -1))
        # 统计 rank 直方（0-based；-1 代表不在 top-k）
        rank_hist = torch.bincount(ranks.clamp(min=-1).add_(1), minlength=K + 1).cpu().tolist()
        # rank_hist[0] 是 -1（未命中）的数量，其余 1..K 是 rank 0..K-1 的数量

        # 5) 抽样打印若干条
        sel = torch.nonzero(m.flatten(), as_tuple=False).flatten()  # 线性坐标
        sel = sel[:min(max_print, sel.numel())]
        bt = torch.stack([sel // T, sel % T], dim=-1)  # [S,2]

        print(f"[KD] M(valid)={n_valid}, hit@{K}={hit_rate * 100:.2f}%, "
              f"mean(|sum(p)-1|)={sum_err_mean:.3e}, top1!=label ratio={(top1_idx != lab).float().mean().item() * 100:.2f}%")
        print(f"[KD] rank hist (bin0=-1=miss, 1..K = rank0..K-1): {rank_hist}")
        for s in bt.cpu().tolist():
            b, t = s
            l = labels[b, t].item()
            k_ids = kd_idx[b, t, :K_print].cpu().tolist()
            k_vals = kd_val[b, t, :K_print].cpu().tolist()
            print(f"  (b={b}, t={t}) label={l} | topK_idx={k_ids} | topK_val={['%.3f' % v for v in k_vals]}")

    def _reduce_supervised_loss(
            self,
            per_token_loss: torch.Tensor,
            token_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce CE without letting long answers dominate every update.

        token_mean preserves the historical objective. sample_mean first
        normalizes each example by its own supervised-token count. hybrid is a
        convex combination and is the safer default for continual VLM SFT.
        """
        token_denom = token_weights.sum().clamp_min(1.0)
        token_mean = (per_token_loss * token_weights).sum() / token_denom

        reduction = str(getattr(self.cfg, "loss_reduction", "token_mean"))
        if reduction == "token_mean":
            return token_mean

        sample_denom = token_weights.sum(dim=1)
        valid_samples = sample_denom > 0
        if not valid_samples.any():
            return token_mean
        sample_losses = (
            (per_token_loss * token_weights).sum(dim=1)
            / sample_denom.clamp_min(1.0)
        )
        sample_mean = sample_losses[valid_samples].mean()
        if reduction == "sample_mean":
            return sample_mean
        if reduction == "hybrid":
            alpha = float(getattr(self.cfg, "sample_mean_alpha", 0.75))
            return alpha * sample_mean + (1.0 - alpha) * token_mean
        raise ValueError(f"Unsupported loss_reduction={reduction!r}")

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None,
                vision_feats: torch.Tensor | None = None,  # [B, Nv, D_v]
                vision_mask: torch.Tensor | None = None,  # [B, Nv]，1=valid
                global_pos: torch.Tensor | None = None,       # [B,Nv]   or [B,V,Nv]
                global_off: torch.Tensor | None = None,       # [B,Nv,4] or [B,V,Nv,4]
                is_causal: bool = True,
                past_states: Optional[List[dict]] = None,
                use_cache: bool = False,
                kd_idx: torch.Tensor | None = None,
                kd_val: torch.Tensor | None = None,
                kd_mask: torch.Tensor | None = None,

                short_idx=None, gold_col=None,
                short_logq: torch.Tensor | None = None,
                enhance_math: bool = True,
                math_sample_mask=None,
                force_checkpoint: bool = False,
                **kwargs,
                ):
        past_key_values = kwargs.pop("past_key_values", None)
        return_dict = kwargs.pop("return_dict", False)
        if past_states is None and past_key_values is not None:
            past_states = past_key_values
        B, T = input_ids.shape
        is_xlong = self.training and (T > getattr(self.cfg, "train_maxlength", T))
        if force_checkpoint:
            is_xlong = True
        if vision_feats is not None:
            if not self.use_vision or self.vision_projector is None:
                raise ValueError("vision_feats was provided but cfg.use_vision=False or projector is None.")
            if kd_idx is not None or kd_val is not None or kd_mask is not None:
                raise NotImplementedError("KD with vision tokens is not implemented yet.")
            if past_states is not None:
                raise NotImplementedError(
                    "vision_feats should only be passed on the first prefill step (past_states=None)."
                )

        if self.training and use_cache and (labels is not None):
            use_cache = False
            warnings.warn("KV-cache should be disabled during training: use_cache has been set to False.")
            past_states = None

        x = self.tok_embed(input_ids)
        T_txt = T

        tok=x
        img_kwargs = {}
        if (vision_feats is not None) and (self.vision_projector is not None):
            if vision_feats.ndim == 4:
                Bv, V, Nv, Dv = vision_feats.shape
                if V != self.num_vision_views:
                    raise ValueError(f"Expected V={self.num_vision_views}, got V={V}")
            elif vision_feats.ndim == 3:
                Bv, Nv, Dv = vision_feats.shape
            else:
                raise ValueError(f"vision_feats must be 3D or 4D, got shape={tuple(vision_feats.shape)}")

            if Bv != B:
                raise ValueError(f"vision_feats batch={Bv} does not match input_ids batch={B}.")

            # 不截断：超了就让 encode 里报错（或你也可以在这先报）
            q_img = self.encode_image_with_qformer(
                vision_feats=vision_feats,
                vision_mask=vision_mask,
                global_pos=global_pos,
                global_off=global_off,
            )
            Bq, Nq, H = q_img.shape
            img_kwargs = {"img_len": int(Nq)}

            # attention_mask / labels 的扩展照旧，只是前缀长度用 Nq


            attn_img = torch.ones((B, Nq), dtype=torch.long, device=input_ids.device)
            attention_mask = torch.cat([attn_img, attention_mask], dim=1)
            x = torch.cat([q_img, tok], dim=1)

            if labels is not None:
                ignore = self.cfg.ignore_index
                labels_exp = torch.full((B, Nq + T_txt), ignore, dtype=labels.dtype, device=labels.device)
                labels_exp[:, Nq:] = labels
                labels = labels_exp
        else:
            x = tok
        x = self.embdrop(x)
        B, L, _ = x.shape
        seq_len = L
        ce_loss = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        kd_ce = torch.zeros((), device=input_ids.device, dtype=torch.float32)

        present_states = [] if use_cache else None
        aux_total = torch.zeros((), device=input_ids.device, dtype=torch.float32)

        for i, layer in enumerate(self.blocks):
            is_moe_layer = (self.cfg.use_moe and (i in self.cfg.moe_layers))

            if is_xlong:
                # 16K 特例：所有非 MoE 层都 checkpoint，最大化省显存
                use_ckpt = (self.training and not use_cache and (not is_moe_layer))
            else:
                # 普通 2K/8K：保持你原来的策略（要 cfg.use_checkpoint=True 且只在偶数层 ckpt）
                use_ckpt = (
                        self.training
                        and self.cfg.use_checkpoint
                        and not use_cache
                        and (i % 2 == 0)
                        and (not is_moe_layer)
                )

            layer_past = past_states[i] if (past_states is not None) else None

            if use_ckpt:
                def layer_fwd(
                        _x,
                        _layer=layer,
                        _layer_past=layer_past,
                        _attn_mask=attention_mask,
                        _is_causal=is_causal,
                ):
                    x_out, aux, _present = _layer(
                        _x,
                        attention_mask=_attn_mask,
                        is_causal=_is_causal,
                        past_state=_layer_past,
                        use_cache=False,  # checkpoint 下不存 cache
                        **img_kwargs,
                    )
                    return x_out, aux

                x, aux_loss = checkpoint(
                    layer_fwd, x,
                    preserve_rng_state=True,
                    use_reentrant=self.cfg.checkpoint_use_reentrant
                )
                present = None
            else:
                x, aux_loss, present = layer(
                    x, attention_mask=attention_mask, is_causal=is_causal,
                    past_state=layer_past, use_cache=use_cache,**img_kwargs

                )

            aux_total = aux_total + aux_loss

            if use_cache:
                present_states.append(present)

        aux_total *= 0.5
        x = self.final_norm(x)  # [B,T,H]

        # ===== 只有推理（无 labels） =====
        if labels is None:
            logits = F.linear(x, self.tok_embed.weight, self.lm_bias)
            unk_id = getattr(self.cfg, "unk_token_id", None)
            if (unk_id is not None) and (0 <= int(unk_id) < logits.size(-1)):
                logits[..., int(unk_id)] = float("-inf")
            if return_dict:
                return CausalLMOutputWithPast(
                    loss=None,
                    logits=logits,
                    past_key_values=present_states if use_cache else None,
                )
            if use_cache:
                return {"logits": logits, "past_states": present_states, "past_key_values": present_states}
            return {"logits": logits}

        if not self.training:
            logits_full = F.linear(x.float(), self.tok_embed.weight.float(), self.lm_bias.float())
            ce_loss = F.cross_entropy(
                logits_full.view(-1, logits_full.size(-1)),
                labels.view(-1).long(),
                ignore_index=self.cfg.ignore_index,
                label_smoothing=0.0,
                reduction="mean",
            )
            loss = ce_loss + aux_total  # eval 下 aux_total≈0，保留无妨
            ret = {"loss": loss, "logits": logits_full, "ce_loss": ce_loss, "aux_loss": aux_total, "kd_loss": kd_ce}
            if use_cache:
                ret["past_states"] = present_states
            return ret

        # ===== 计算主 CE（两条路：adaptive / short-list 与 full） =====
        if short_idx is not None and gold_col is not None:
            # --- 短清单 CE：保持不变 ---
            logits_short = self._project_subset_logits(x, short_idx.long())  # [B,T,Kp]
            logits_short = self._mask_shortlist_dups(short_idx, logits_short)
            logits_short_corr = logits_short
            if short_logq is not None:
                logits_short_corr = logits_short_corr - short_logq.to(logits_short.device, logits_short.dtype)

            ce_logp = F.log_softmax(logits_short_corr, dim=-1)  # [B,T,Kp]
            tgt_col = gold_col.long()  # [B,T]（恒为 0）
            valid = (labels != self.cfg.ignore_index)

            ce_nll = -ce_logp.gather(-1, tgt_col.unsqueeze(-1)).squeeze(-1)  # [B,T]
            ce_loss = self._reduce_supervised_loss(ce_nll, valid.float())
            loss = ce_loss + aux_total

            # KD（对短清单对齐），保持不变
            has_kd = (kd_idx is not None) and (kd_val is not None) and (kd_mask is not None) and (
                        kd_idx.numel() > 0)
            if has_kd:
                kd_alpha = 0.2
                kd_tau = 1.5

                eq = (kd_idx.long().to(x.device).unsqueeze(-1) == short_idx.long().to(x.device).unsqueeze(
                    -2))  # [B,T,K_kd,Kp]
                pos_map = eq.float().argmax(dim=-1)  # [B,T,K_kd]
                hit = eq.any(dim=-1)  # [B,T,K_kd]
                s_logp = F.log_softmax(logits_short / kd_tau, dim=-1)  # [B,T,Kp]
                s_logp_kd = torch.take_along_dim(s_logp, pos_map, dim=-1)  # [B,T,K_kd]
                t_prob = torch.softmax(kd_val.to(x.device) / kd_tau, dim=-1)  # [B,T,K_kd]
                hit = hit.bool() if hit.dtype != torch.bool else hit

                s_logp_kd = s_logp_kd * hit.float()
                t_prob = t_prob * hit.float()
                denom = t_prob.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                t_prob = t_prob / denom
                kd_token = -(t_prob * s_logp_kd).sum(dim=-1)
                mask_tok = kd_mask.bool() & valid.bool() & hit.any(dim=-1)
                if mask_tok.any():
                    kd_ce = kd_token[mask_tok].mean()
                    kd_ce = (kd_tau * kd_tau) * kd_ce * kd_alpha
                    kd_ce = kd_ce.to(torch.float32)
                    loss = loss + kd_ce

            self._train_step += 1
            if use_cache:
                return {"loss": loss, "past_states": present_states}
            return {"loss": loss, "ce_loss": ce_loss, "aux_loss": aux_total, "kd_loss": kd_ce}

        # ===== 全量 CE 分支（这里加入“高损失截尾”，仅作用于 CE） =====
        logits_full = None
        is_xlong = (T > getattr(self.cfg, "train_maxlength", T))  # 例如 train_maxlength=8192 时，T=16384 就视为 xlong
        if self.training and is_xlong and (vision_feats is not None):
            raise RuntimeError(
                f"[TinyLLM] xlong training (T={T}) with vision_feats is not supported yet. "
                "当前实现的 xlong 分块 CE 没有正确处理 vision 前缀，"
                "请减小文本长度到 train_maxlength 以下，或者在 xlong 训练时关闭 vision_feats。"
            )
        if self.training:
            if is_xlong:
                # --- 16K 等超长序列：分块算 CE，避免一次性 [B,T,V] ---
                chunk_size = 2048  # 你可以试 2048/4096，自行权衡
                per_loss_chunks = []
                W = self.tok_embed.weight
                b = self.lm_bias

                for start in range(0, T, chunk_size):
                    end = min(T, start + chunk_size)
                    x_slice = x[:, start:end, :]  # [B, c, H]
                    labels_slice = labels[:, start:end]  # [B, c]

                    # 如果这一块全是 ignore_index，就直接给 0，省点算力
                    if not (labels_slice != self.cfg.ignore_index).any():
                        loss_slice = x_slice.new_zeros(
                            (B, end - start),
                            dtype=torch.float32
                        )
                    else:
                        logits_slice = F.linear(
                            x_slice.float(),  # 为了数值稳定，仍然用 float32 算
                            W.float(),
                            b.float() if b is not None else None,
                        )  # [B, c, V]

                        loss_slice = F.cross_entropy(
                            logits_slice.view(-1, logits_slice.size(-1)),
                            labels_slice.contiguous().view(-1).to(torch.long),
                            ignore_index=self.cfg.ignore_index,
                            label_smoothing=0.0,
                            reduction="none",
                        ).view(B, end - start)  # [B, c]

                    per_loss_chunks.append(loss_slice)

                per_loss = torch.cat(per_loss_chunks, dim=1)  # [B, T]
                # 注意：这里不再设置 logits_full，后面训练分支也不会返回 logits，节省显存
                logits_full = None
            else:
                # --- 普通 2K/8K 序列：保持你原来的实现 ---
                logits_full = F.linear(
                    x.float(), self.tok_embed.weight.float(), self.lm_bias.float()
                )  # [B,T,V]
                per_loss = F.cross_entropy(
                    logits_full.view(-1, logits_full.size(-1)),
                    labels.view(-1).to(torch.long),
                    ignore_index=self.cfg.ignore_index,
                    label_smoothing=0.0,
                    reduction="none"
                ).view(B, seq_len)  # 每 token NLL

            valid_bool = (labels != self.cfg.ignore_index).bool()

            # 先做 end_text 降权（仅在 valid 位）
            w = valid_bool.float()
            if hasattr(self, "end_text_tok_id"):
                eos_mask = (labels == self.end_text_tok_id) & valid_bool
                # w = torch.where(eos_mask, w.new_full((), float(self.eso_loss_radio)), w)

            # === 损失截尾（百分位，默认 warmup 后丢 top 1% 高损失 token）===
            trunc_ratio = self.drop_high_loss  # e.g., 0.01 = 丢 top 1%

            min_tokens_for_trunc = getattr(self.cfg, "loss_trunc_min_tokens", 100)  # 至少多少有效 token 才截
            if valid_bool.any() and (trunc_ratio > 0.0):
                v_losses = per_loss[valid_bool]
                if v_losses.numel() >= int(min_tokens_for_trunc):
                    # 找到“要丢弃的 top-k 中最小的那个值”，作为阈值
                    num_trunc = max(1, int(v_losses.numel() * trunc_ratio))
                    top_vals = torch.topk(v_losses, num_trunc, largest=True, sorted=False).values
                    loss_threshold = top_vals.min()
                    # 保留：小于等于阈值的，或无效位（pad/ignore 仍为 0 权重）
                    keep_mask = (per_loss <= loss_threshold) | (~valid_bool)
                    w = w * keep_mask.float()
            loss_class_weight = getattr(self, "class_w_mathish", None)
            if loss_class_weight is not None and enhance_math:
                # w: 之前已经包含了 valid_mask、end_text 降权、截尾等信息
                # 这里我们再乘一个 per-token 的数学权重 cw，以及可选的样本级 boost。

                # ---- 1) 先构造 “label → 数学权重” 映射（只在 valid 位生效）----
                cw = torch.ones_like(per_loss, dtype=per_loss.dtype, device=per_loss.device)

                idx = labels[valid_bool].to(torch.long)  # 这些位置的 label 一定不是 ignore_index
                base_cw_valid = loss_class_weight.to(per_loss.dtype).to(per_loss.device).index_select(0, idx)
                # base_cw_valid: [num_valid]，>1 的那些就是你选出来的 mathish token

                if math_sample_mask is None:
                    # ★ 兼容老逻辑：没传 mask，就当整 batch 都是数学样本
                    # → 所有 valid token 均按 base_cw_valid 加权（和原来完全一致）
                    cw_valid = base_cw_valid
                else:
                    # ★ 新逻辑：只有 isMath=1 的样本才启用 per-id 数学权重
                    # math_sample_mask: [B]，0/1
                    gate = math_sample_mask.view(-1, 1).to(per_loss.dtype)  # [B,1]
                    gate_full = gate.expand_as(per_loss)  # [B,T]
                    gate_valid = gate_full[valid_bool]  # [num_valid]

                    # gate=0 → cw=1（非数学样本不加权）
                    # gate=1 → cw=base_cw_valid（数学样本里才用 per-id mathish 权重）
                    cw_valid = 1.0 + gate_valid * (base_cw_valid - 1.0)

                cw[valid_bool] = cw_valid
                w = w * cw

                # ---- 2) （可选）再给 “整道数学样本” 一个样本级 boost，比如 ×1.2 ----
                if math_sample_mask is not None:
                    SAMPLE_BOOST = 1.0  # 如果你觉得太猛可以改成 1.1
                    sample_gate = 1.0 + (SAMPLE_BOOST - 1.0) * gate_full  # [B,T]
                    # ignore 位置本来 w=0，乘多少还是 0，不会出问题
                    w = w * sample_gate

            ce_loss = self._reduce_supervised_loss(per_loss, w)

        loss = ce_loss + aux_total

        # ====== KD（可选，保持不变）======
        has_kd = (kd_idx is not None) and (kd_val is not None) and (kd_mask is not None)
        if has_kd:
            kd_alpha = 0.2
            kd_tau = 1.5

            kd_idx = kd_idx.to(x.device, dtype=torch.long)
            take_btK = self._project_subset_logits(x.float(), kd_idx)  # [B,T,K]

            t_prob = torch.softmax(kd_val.to(take_btK.device) / kd_tau, dim=-1)  # [B,T,K]
            s_logp = torch.log_softmax(take_btK / kd_tau, dim=-1)

            mask_tok = kd_mask.to(take_btK.device).bool()
            mask_tok = mask_tok & (labels != self.cfg.ignore_index)

            if mask_tok.any():
                kd_token = -(t_prob * s_logp).sum(dim=-1)
                kd_ce = kd_token[mask_tok].mean()
                kd_ce = (kd_tau * kd_tau) * kd_ce * kd_alpha
                kd_ce = kd_ce.to(torch.float32)
                loss = loss + kd_ce

        self._train_step += 1

        # ===== 返回 =====
        if return_dict:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits_full,
                past_key_values=present_states if use_cache else None,
            )
        if use_cache:
            ret = {"loss": loss, "past_states": present_states}
            if logits_full is not None:
                ret["logits"] = logits_full
            return ret

        if logits_full is not None:
            return {"loss": loss, "logits": logits_full, "ce_loss": ce_loss, "aux_loss": aux_total,
                    "kd_loss": kd_ce}

        return {"loss": loss, "ce_loss": ce_loss, "aux_loss": aux_total, "kd_loss": kd_ce}

    def _wrap_linear_with_lora(
            self,
            module: nn.Module,
            attr: str,
            adapter_name: str,
            rank: int = 16,
            dropout: float = 0.05,
            alpha: float = 16.0,
            mode: str = "exclusive",
            cap_norm: float | None = 1.0,
            global_scale: float = 1.0,
    ):
        base = getattr(module, attr, None)
        if base is None:
            return
        # Existing wrappers must register the new adapter as well; otherwise
        # switching between ARC and OPD on one resident base is impossible.
        if isinstance(base, LoraLinear):
            if adapter_name not in base.adapters:
                base.register_adapter(name=adapter_name, rank=rank, dropout=dropout, alpha=alpha)
            return

        lora = LoraLinear(base, mode=mode, cap_norm=cap_norm, global_scale=global_scale)
        lora.register_adapter(
            name=adapter_name,
            rank=rank,
            dropout=dropout,
            alpha=alpha,
        )
        lora.activate(adapter_name, exclusive=True)
        setattr(module, attr, lora)

    def attach_lora_adapter(
            self,
            adapter_name: str,
            rank: int = 16,
            dropout: float = 0.05,
            alpha: float = 16.0,
            target: str = "attn_mlp_top_half",  # 你可以扩展其它策略
    ):
        """
        在指定层位上挂一个名为 adapter_name 的 LoRA。
        可以多次调用，用不同 adapter_name 做多个 LoRA。
        """
        L = len(self.blocks)
        if target == "attn_mlp_top_half":
            start = L // 2
            end = L
        elif target == "all":
            start, end = 0, L
        elif target == "attn_mlp_skip_first4":
            # 跳过最底下 4 层，从第 4 层开始一路到顶
            # （0-based：0,1,2,3 不挂；4,...,L-1 都挂）
            start = min(4, L)  # 防止 L < 4 的极端情况
            end = L
        else:
            # 你可以自己扩展其他策略
            start, end = 0, L

        for i in range(start, end):
            block = self.blocks[i]

            # 1) 注意力（非 SSM block 才有 selfatt）
            if hasattr(block, "selfatt"):
                attn = block.selfatt
                for attr in ["w_q", "w_k", "w_v", "w_o"]:
                    self._wrap_linear_with_lora(
                        attn, attr,
                        adapter_name=adapter_name,
                        rank=rank,
                        dropout=dropout,
                        alpha=alpha,
                        mode="exclusive",
                        cap_norm=1.0,
                        global_scale=1.0,
                    )

            if isinstance(block.MLP, MLP):
                mlp = block.MLP
                for attr in ["upsamp", "downsamp", "swiGate"]:
                    if hasattr(mlp, attr):
                        self._wrap_linear_with_lora(
                            mlp, attr,
                            adapter_name=adapter_name,
                            rank=rank,
                            dropout=dropout,
                            alpha=alpha,
                            mode="exclusive",
                            cap_norm=1.0,
                            global_scale=1.0,
                        )
            # MoE / shared expert 先不用挂，之后要的话再细化

    def set_active_lora(self, names: list[str] | None):
        """
        全局设置 LoRA 激活的 adapter 列表。
        - names 为 None 或 [] 表示关闭所有 LoRA。
        - 多个名字配合 LoraLinear.mode="additive"/"weighted" 使用。
        """
        for m in self.modules():
            if isinstance(m, LoraLinear):
                if not names:
                    m.active = []
                else:
                    # Different adapters may target different layer ranges.
                    # Activate only names actually registered on each wrapper.
                    m.set_active([name for name in names if name in m.adapters])

    def activate_single_lora(self, name: str | None):
        if name is None:
            self.set_active_lora([])
        else:
            self.set_active_lora([name])

    def get_lora_state_dict(self, adapter_name: str | None = None):
        """
        只导出 LoRA adapter 的参数（可选指定某一个 adapter）。
        返回的 key 仍然是模型里的完整路径，方便直接 load_state_dict(strict=False)。
        """
        full = self.state_dict()
        out = {}
        for k, v in full.items():
            if ".adapters." not in k:
                continue
            if adapter_name is not None and f".adapters.{adapter_name}." not in k:
                continue
            out[k] = v.detach().cpu()
        return out
    def load_lora_state_dict(
        self,
        state: dict,
        adapter_name: str | None = None,
        strict: bool = False,
    ):
        """
        只加载 LoRA adapter 的参数（和 get_lora_state_dict 对称）。

        - state 通常来自 torch.load(lora_xxx.pt)，里边只有 LoRA 的 key。
        - adapter_name 仅用于过滤对应 adapter；如果为 None 则加载所有 ".adapters." 的参数。
        """
        if adapter_name is not None:
            filtered = {
                k: v for k, v in state.items()
                if f".adapters.{adapter_name}." in k
            }
        else:
            filtered = {
                k: v for k, v in state.items()
                if ".adapters." in k
            }

        # 注意 strict=False：我们只想更新已有的 LoRA 权重，其他缺失参数无所谓
        missing, unexpected = self.load_state_dict(filtered, strict=strict)

        if strict:
            if missing:
                print(f"[LoRA-load] missing[:10]   = {missing[:10]} (total {len(missing)})")
            if unexpected:
                print(f"[LoRA-load] unexpected[:10]= {unexpected[:10]} (total {len(unexpected)})")

        return missing, unexpected

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[Cache] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            vision_feats: torch.Tensor | None = None,
            vision_mask: torch.Tensor | None = None,
            global_pos: torch.Tensor | None = None,
            global_off: torch.Tensor | None = None,
            **kwargs,
    ):
        # 增量步：只喂最后一个 token，并清掉视觉相关输入
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_ids.shape, dtype=torch.long)
            else:
                attention_mask = attention_mask[:, -input_ids.size(1):]  # T 对齐

            vision_feats = None
            vision_mask = None
            global_pos = None
            global_off = None

        # prefill：必须把 global_pos/global_off 原样传下去，否则 forward 收不到
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "vision_feats": vision_feats,
            "vision_mask": vision_mask,
            "global_pos": global_pos,
            "global_off": global_off,
            "use_cache": True,
        }

    def _prepare_generation_config(
            self,
            generation_config: GenerationConfig | None = None,
            model_kwargs: dict | None = None,
            **kwargs,
    ):
        """
        自己接管 GenerationMixin 的 config 准备逻辑，避免去碰
        self.config._get_non_default_generation_parameters / _from_model_config
        这些我们没实现的接口。
        """
        # 1) 合并 model_kwargs 和裸 kwargs
        if model_kwargs is None:
            model_kwargs = {}
        merged_kwargs = {**model_kwargs, **kwargs}

        # 2) 拿一个 base GenerationConfig
        if generation_config is None:
            base = getattr(self, "generation_config", None)
            if isinstance(base, GenerationConfig):
                generation_config = copy.deepcopy(base)
            else:
                generation_config = GenerationConfig(
                    eos_token_id=getattr(self.cfg, "eos_token_id", None),
                    pad_token_id=getattr(self.cfg, "pad_token_id", None),
                )
        else:
            # 允许传 dict / GenerationConfig / 其它 config
            if isinstance(generation_config, dict):
                generation_config = GenerationConfig(**generation_config)
            elif isinstance(generation_config, GenerationConfig):
                generation_config = copy.deepcopy(generation_config)
            else:
                # 大概率是一个 PretrainedConfig；我们不上 _from_model_config，防止再踩坑
                try:
                    generation_config = GenerationConfig.from_model_config(generation_config)
                except Exception:
                    generation_config = GenerationConfig(
                        eos_token_id=getattr(self.cfg, "eos_token_id", None),
                        pad_token_id=getattr(self.cfg, "pad_token_id", None),
                    )

        # 3) 用 generation_config 自己的字段当作“合法生成参数”列表
        try:
            gen_param_keys = set(generation_config.to_dict().keys())
        except Exception:
            gen_param_keys = set(vars(generation_config).keys())

        cleaned_model_kwargs: dict = {}
        for k, v in merged_kwargs.items():
            if k in gen_param_keys:
                # 比如 max_new_tokens / do_sample / eos_token_id / pad_token_id / num_beams ...
                setattr(generation_config, k, v)
            else:
                # 留下真正要传给 forward 的，比如 input_ids / attention_mask / past_key_values
                cleaned_model_kwargs[k] = v

        return generation_config, cleaned_model_kwargs


