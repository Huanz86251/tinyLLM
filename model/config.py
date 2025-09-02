from transformers import PretrainedConfig
from transformers import AutoTokenizer

class Config(PretrainedConfig):
    model_type = "tinyLLM"

    def __init__(self,
                vocab_size=20000,
                hidden_size=640,
                num_hidden_layers=21,
                num_attention_heads=10,
                num_key_value_heads=5,
                max_position_embeddings=16384,
                RoPE_base=2e4,
                RoPE_NTK=False,
                dropout=0.1,
                rms_norm_eps=1e-5,
                qkrms_norm_eps=1e-5,
                use_qk_norm=True,
                kv_cache_dtype="auto",
                use_moe=True,
                adapter_type="none",
                use_swiGLU=True,
                bos_token_id=None,
                eos_token_id=0,
                unk_token_id=3,
                pad_token_id=0,
                embeddingdropout=0.0,
                learnable_temp=True,
                use_affine=True,
                use_sampleattion=False,
                train_maxlength=2048,
                 mlp_ratio=4.0,
                 ignore_index=-100,
                 drop_path=0.05,
                 residual_dropout=0.05,
                 moe_layers=[5, 10, 15, 19],
                 num_expert=4,
                 moe_use_detach=False,
                 moe_cap_factor=1.5,
                 moe_aux_weight=5.0,
                 use_checkpoint=False,
                 checkpoint_use_reentrant=True,
                **kwargs):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id,unk_token_id=unk_token_id,pad_token_id=pad_token_id,**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers         #transformer层数
        self.num_attention_heads = num_attention_heads     #q head数
        self.num_key_value_heads = num_key_value_heads     #kv head数，q head % k head=0
        self.max_position_embeddings = max_position_embeddings # 理论上下文极限值，已达到tokenizer 上限，扩展需修改tokenizer
        self.RoPE_base=RoPE_base                           # RoPE时频率相关theta base大小，越大支持上下文越多，目前支持最大4k，1e6可支持16k，但针对短文本敏感降低
        self.RoPE_NTK=RoPE_NTK                             #RoPE时是否开启动态NTK扩展当前base 4k的能力到16k，一般训练可关闭
        self.dropout = dropout                             #控制attention和MLP dropout概率
        self.train_maxlength=train_maxlength               #训练时文本最大长度4096
        self.rms_norm_eps = rms_norm_eps                   #非attention内部rms norm时计算rms时算完平均值后在rsqrt前增加的值
        self.qkrms_norm_eps = qkrms_norm_eps               # attention内部在qk投影后rms norm时计算rms时算完平均值后在rsqrt前增加的值
        self.use_qk_norm = use_qk_norm                      #是否启用qk norm
        self.kv_cache_dtype = kv_cache_dtype
        self.use_moe = use_moe
        self.adapter_type = adapter_type
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
        self.embeddingdropout=embeddingdropout          #embedding dropout概率，默认0，最大0.05
        self.learnable_temp= learnable_temp             #是否对qkrms_norm启用一个可学习的增幅
        self.use_affine=use_affine                      #是否对rms_norm启用一个可学习的增幅
        self.use_sampleatt=use_sampleattion             #是否使用自己写的attention，效率低，无cuda优化
        self.use_swiGLU=use_swiGLU                      #MLP是否使用swiGLU，False使用up-silu-down
        self.mlp_ratio=mlp_ratio                        #MLP参数量2*mlp_ratio*H^2
        self.ignore_index=ignore_index
        self.drop_path=drop_path                        #attention MLP 残差分支最大droppath的概率,随着层数线性增长至最大值
        self.residual_dropout=residual_dropout          #attention MLP 残差分支逐元素drop的概率
        self.use_moe=use_moe
        self.moe_layers=moe_layers                      #moe加载在哪些层
        self.num_expert=num_expert
        self.moe_use_detach=moe_use_detach              # false让主损失的梯度回到moe gate，训练不稳但能真的帮助gate学会使用不同的expert，否则gate趋于平均分布
        self.moe_cap_factor=moe_cap_factor              #1-1.5之间选择，决定每个expert加载的最大token量
        self.moe_aux_weight=moe_aux_weight
        self.use_checkpoint = False                     #开启训练慢降显存
        self.checkpoint_use_reentrant = True