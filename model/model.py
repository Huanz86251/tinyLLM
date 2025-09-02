import math
from os.path import split

from werkzeug.debug.repr import missing
from typing import Dict, List, Optional, Literal
from model.config import Config

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.utils.checkpoint import checkpoint

def gumbel_noise(x:torch.Tensor):
    #-log(-log(U))
    u=torch.rand_like(x).clamp(1e-9,1-1e-9)
    return -torch.log(-torch.log(u))
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
        self.learnable_temp=learnable_temp
        self.q_norm=RMSNorm(head_dim,eps)
        self.k_norm=RMSNorm(head_dim,eps)
        if self.learnable_temp:
            self.log_alpha=nn.Parameter(torch.zeros(num_head))
        else:
            self.register_parameter("log_alpha",None)
    def forward(self,q:torch.Tensor,k:torch.Tensor):
        q=self.q_norm(q)
        k=self.k_norm(k)
        if self.learnable_temp:

            alpha=self.log_alpha.exp().to(device=q.device,dtype=q.dtype)
            q=q*alpha.view(1,-1,1,1)
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
                 use_NTK:bool=False,
                 train_length:int=4096
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
        self.RoPE=RoPE(self.dim_perhead,max_position=max_position_embeddings,base=RoPE_base,use_NTK=use_NTK,train_length=train_length)
        if self.use_qk_RMSnorm:
            self.qk_RMSnorm=QK_RMSNorm(self.dim_perhead,self.num_heads,self.qk_RMSeps)
        else:
            self.qk_RMSnorm=None

        self.use_sample_attention=use_sample_attention
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

    def forward(self,x:torch.Tensor,attention_mask:torch.Tensor|None=None,is_causal:bool=True,past_kv:tuple[torch.Tensor,torch.Tensor] | None=None,use_cache:bool=False):
        B,T,C=x.shape
        q=self.w_q(x)
        k=self.w_k(x)
        v=self.w_v(x)

        q=self._reshape(q,self.num_heads,self.dim_perhead)
        k=self._reshape(k,self.num_kv_heads,self.dim_perhead)
        v=self._reshape(v,self.num_kv_heads,self.dim_perhead)

        if self.use_qk_RMSnorm:
            q,k=self.qk_RMSnorm(q,k)
        past_T=past_kv[0].size(-2) if past_kv is not None else 0
        q=self.RoPE(q,start_pos=past_T)
        k=self.RoPE(k,start_pos=past_T)
        TotalT = T
        if past_kv is not None: #kv cache 合并
           past_k,past_v=past_kv
           k=torch.cat([past_k,k],dim=-2)
           v=torch.cat([past_v,v],dim=-2)
           TotalT=k.size(-2)

        k_kv, v_kv = k, v
        if self.num_kv_heads!=self.num_heads:
            repeat=self.num_heads//self.num_kv_heads
            k=k.unsqueeze(2).expand(B,self.num_kv_heads,repeat,TotalT,self.dim_perhead)
            k=k.reshape(B,self.num_heads,TotalT,self.dim_perhead)
            v=v.unsqueeze(2).expand(B,self.num_kv_heads,repeat,TotalT,self.dim_perhead)
            v=v.reshape(B,self.num_heads,TotalT,self.dim_perhead)
        if attention_mask is not None:
            key_pad = ~attention_mask.to(torch.bool)
            attention_mask = key_pad[:, None, None, :]
        if self.use_sample_attention:
            y=self.sample_attention(q,k,v,attention_mask,is_causal,self.dropout)
        else:
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=attention_mask,dropout_p=self.dropout if self.training else 0.0,is_causal=is_causal)
        y=y.transpose(1,2).contiguous().view(B,T,self.hidden_size)
        return self.w_o(y),((k_kv, v_kv) if use_cache else None)

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
class Top1Router(nn.Module):
    def __init__(self,hidden_size:int,num_expert:int,enable_detach:bool=False):
        super().__init__()
        self.decider=nn.Linear(hidden_size,num_expert,bias=True)
        self.num_expert=num_expert
        self.enableDetach = enable_detach
    def forward(self,x:torch.Tensor):
        if self.enableDetach:
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
    def __init__(self,hidden_dim:int,dropout,num_expert:int,amplify:float=4,use_swiGLU:bool=True,cap_factor:float=1.25,enable_detach:bool=False):
        super().__init__()
        self.num_expert=num_expert
        self.expert=nn.ModuleList([MLP(hidden_dim,dropout,amplify,use_swiGLU) for i in range(num_expert)])
        self.fallback=MLP(hidden_dim,dropout,amplify,use_swiGLU)
        self.router=Top1Router(hidden_dim,num_expert,enable_detach)
        self.cap_factor=cap_factor
        self.enableDetach=enable_detach
        self.register_buffer("top2_alpha", torch.tensor(0.0))
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
                y_kept=y_kept*p_kept
                sorted1_y[sj:rj]=y_kept
            if ej>rj:
                x_leak=sorted1_x[rj:ej]
                y_leak=self.fallback(x_leak)
                sorted1_y[rj:ej]=y_leak

        inv1_idx=torch.argsort(e1_idx)
        y1=sorted1_y[inv1_idx].view(B,T,H)
        e2_orderd,e2_idx=e2.sort()
        x2=x[e2_idx]
        sorted2_p=w2[e2_idx]
        if self.enableDetach:
            sorted2_p=sorted2_p.detach()
        count2=torch.bincount(e2_orderd, minlength=self.num_expert)
        prefix2=torch.cumsum(count2, dim=0)
        start2=prefix2-count2
        end2=prefix2
        y2_sorted=torch.zeros_like(x2)
        for j in range(self.num_expert):
            sj, ej = start2[j],end2[j]
            if ej==sj:
                continue
            y2_sorted[sj:ej] = self.expert[j](x2[sj:ej]) * sorted2_p[sj:ej]

        inv2=torch.argsort(e2_idx)
        y2=y2_sorted[inv2]   # [N,H]
        y2=y2.view(B,T,H)
        y=(y1+y2)
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
                 use_ROPE_NTK:bool=True,
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
                 moe_num_expert:int=4
                 ):
        super().__init__()
        self.register_buffer("_env",torch.empty(0))
        self.gamma_att = nn.Parameter(torch.ones(1) * 1e-3)
        self.gamma_mlp = nn.Parameter(torch.ones(1) * 1e-3)
        if qkrms_eps is None:
            self.qkrms_eps=rms_eps
        else:
            self.qkrms_eps = qkrms_eps
        if num_kv_heads is None:
            self.num_kv_heads=num_heads
        else:
            self.num_kv_heads= num_kv_heads
        self.attRMS_norm=RMSNorm(hidden_size,rms_eps,use_affine,self._env.dtype,self._env.device)
        self.selfatt=Mutihead_attention(hidden_size,num_heads,self.num_kv_heads,max_position_embeddings,RoPE_base,dropout,use_qk_RMSnorm,self.qkrms_eps,learnable_temp,use_sampleatt,use_ROPE_NTK,training_length)
        self.MLP=MLP(hidden_size,dropout,mlp_ratio,use_swiGLU)
        self.is_moe = use_moe_layer
        self.gamma_min = 0.2
        self.gamma_max = 1.0
        if self.is_moe:
            self.MLP = MoEMLP(hidden_size,dropout,moe_num_expert,mlp_ratio,use_swiGLU,moe_cap_factor,moe_use_detach)
        else:
            self.MLP = MLP(hidden_size, dropout, mlp_ratio, use_swiGLU)
        self.droppath=Drop_path(drop_path)
        self.MLPrms_norm=RMSNorm(hidden_size,rms_eps,use_affine,self._env.dtype,self._env.device)
        self.residual_dropout=nn.Dropout(resid_dropout)

    def _bounded(self, raw):
        # γ = γ_min + (γ_max-γ_min) * sigmoid(raw)
        return self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(raw)
    def forward(self,x:torch.Tensor,attention_mask:torch.Tensor|None=None,is_causal:bool=True,past_key_value:tuple[torch.Tensor,torch.Tensor]|None=None,use_cache:bool=False):
        attout,past_key_value=self.selfatt(self.attRMS_norm(x),attention_mask,is_causal,past_key_value,use_cache)
        attout=attout*self._bounded(self.gamma_att)
        x=x+self.droppath(self.residual_dropout(attout))
        if self.is_moe and self.training:
            mlp_out,aux_loss=self.MLP(self.MLPrms_norm(x))
            mlp_out=mlp_out*self._bounded(self.gamma_mlp)
        else:
            normed = self.MLPrms_norm(x)#防止checkpoint因为奇怪标量报错
            mlp_out= self.MLP(self.MLPrms_norm(x))
            mlp_out=mlp_out*self._bounded(self.gamma_mlp)
            aux_loss = normed.sum() * 0.0
        x=x+self.droppath(self.residual_dropout(mlp_out))
        return x,aux_loss,(past_key_value if use_cache else None)
class TinyLLM(nn.Module):
    def __init__(self,cfg:Config):
        super().__init__()
        self.cfg = cfg
        self.end_text_tok_id=cfg.eos_token_id
        hidden_size=cfg.hidden_size
        vocab_size=cfg.vocab_size
        self.tok_embed=nn.Embedding(vocab_size,hidden_size)
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.02)
        self.embdrop=nn.Dropout(cfg.embeddingdropout)
        droppath_list=[(cfg.drop_path*i/max(cfg.num_hidden_layers-1,1)) for i in range( cfg.num_hidden_layers)]
        self.blocks=nn.ModuleList([Transformer_block(
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
        use_ROPE_NTK=cfg.RoPE_NTK,
        training_length=cfg.train_maxlength,
        use_swiGLU=cfg.use_swiGLU,
        mlp_ratio=cfg.mlp_ratio,
        RoPE_base=cfg.RoPE_base,
        max_position_embeddings=cfg.max_position_embeddings,
        drop_path = droppath_list[i],

        resid_dropout=cfg.residual_dropout,
        use_moe_layer=(getattr(cfg, "use_moe", False) and (i in cfg.moe_layers)),
        moe_use_detach = cfg.moe_use_detach,
        moe_cap_factor= cfg.moe_cap_factor,
        moe_num_expert= cfg.num_expert

        ) for i in  range(cfg.num_hidden_layers)])
        self.register_buffer("_env0",torch.empty(0))
        self.final_norm=RMSNorm(cfg.hidden_size,cfg.rms_norm_eps,cfg.use_affine,self._env0.dtype,self._env0.device)
        self.lm_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self,input_ids:torch.Tensor,attention_mask: torch.Tensor | None = None,labels: torch.Tensor | None = None,is_causal: bool = True,past_key_value:list[tuple[torch.Tensor,torch.Tensor]]|None=None,use_cache:bool=False):
        B,T=input_ids.shape
        if self.training and use_cache:
            use_cache=False
            warnings.warn("KV-cache should be disabled during training: use_cache has been set to False.")
            past_key_value = None
        x=self.tok_embed(input_ids)
        x=self.embdrop(x)
        present_key_values = [] if use_cache else None
        aux_total=x.new_zeros(())
        for i,layer in enumerate(self.blocks):
            is_moe_layer = (self.cfg.use_moe and (i in self.cfg.moe_layers))
            use_ckpt = (self.training and self.cfg.use_checkpoint and not use_cache and i%2==0 and (not is_moe_layer))
            pastkv = past_key_value[i] if (past_key_value is not None) else None
            if use_ckpt:
                def layer_fwd(_x):

                    x_out, aux, _present = layer(_x, attention_mask, is_causal, pastkv, use_cache=False)
                    return x_out, aux

                x, aux_loss = checkpoint(
                    layer_fwd, x,
                    preserve_rng_state=True,  # 保证 Dropout/Gumbel 在重算时“抽到同样的随机数”
                    use_reentrant=self.cfg.checkpoint_use_reentrant
                )
                present = None
            else:

                x,aux_loss,present=layer(x,
                             attention_mask=attention_mask,is_causal=is_causal,past_key_value=pastkv,use_cache=use_cache)

            aux_total+=aux_loss
            if use_cache:
                present_key_values.append(present)
        x=self.final_norm(x)
        logits = F.linear(x, self.tok_embed.weight, self.lm_bias)



        if labels is None:
            logits[...,self.cfg.unk_token_id]=float("-inf")
            if use_cache:
                return {"logits": logits, "past_key_values":present_key_values}
            return {"logits":logits}

        if self.cfg.use_moe:
            aux_total=aux_total/len(self.cfg.moe_layers)*self.cfg.moe_aux_weight
        labels=labels.masked_fill(labels==self.cfg.unk_token_id,self.cfg.ignore_index).to(dtype=torch.long)
        per_loss=F.cross_entropy(logits.view(-1,logits.size(-1)),labels.view(-1),ignore_index=self.cfg.ignore_index,label_smoothing=0.00,reduction="none").view(B, T)
        valid = (labels!=self.cfg.ignore_index).float()
        w = valid
        w = torch.where(labels ==self.end_text_tok_id, w.new_full((), 0.1), w)
        denom = w.sum().clamp_min(1.0)
        ce_loss= (per_loss * w).sum() / denom
        #print("ce",ce_loss)
        loss=ce_loss+aux_total
        #print("loss",loss)

        if use_cache:
            return {"logits": logits, "loss": loss,"past_key_values":present_key_values}
        return {"logits": logits, "loss": loss}











