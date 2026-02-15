from transformers import PretrainedConfig
from transformers import AutoTokenizer

class Config(PretrainedConfig):
    model_type = "tinyLLM"

    def __init__(self,
                 eos_token_id: int | None = 73440,
                 pad_token_id: int | None = 73440,
                use_tcn = False,
                tcn_layers = [],
                 num_query_tokens=128,
                vocab_size=73448,
                hidden_size=1280,
                num_hidden_layers=24,
                num_attention_heads=20,
                num_key_value_heads=4,
                max_position_embeddings=8192,
                RoPE_base=1e4,
                rope_type="yarn",#["default","yarn","dynamic"]
                dropout=0.00,
                rms_norm_eps=1e-5,
                qkrms_norm_eps=1e-5,
                use_qk_norm=True,
                kv_cache_dtype="auto",
                use_moe=False,
                adapter_type="none",
                use_swiGLU=True,
                bos_token_id=None,
                use_ssm=False,
                ssm_layers=[],#[4,11,19]
                embeddingdropout=0.0,
                learnable_temp=True,
                use_affine=True,
                use_sampleattion=False,
                eso_loss_radio=1.0,
                train_maxlength=8192,
                drop_high_loss=0.0,
                mlp_ratio_front: float = 3.5,
                mlp_ratio_mid: float = 4.0,
                mlp_ratio_back: float = 4.5,
                mlp_mid_start: int | None = 8,  # 中段起始层（含），0-index
                mlp_back_start: int | None = 16,  # 后段起始层（含），0-index
                mlp_ratio_overrides: dict | None = {21: 5.0,22: 5.0, 23: 5.0},
                mlp_ratio=4.5,
                ignore_index=-100,
                loss_reduction: str = "token_mean",
                sample_mean_alpha: float = 0.75,
                drop_path=0.00,
                residual_dropout=0.00,
                moe_layers=[],
                num_expert=4,
                moe_use_detach=False,
                moe_cap_factor=1.5,
                moe_aux_weight=5.0,
                use_checkpoint=False,
                checkpoint_use_reentrant=False,
                use_adaptive_softmax: bool = True,
                adaptive_cutoffs = [20000, 60000],
                 vision_dropout=0.0,
                adaptive_div: float = 4.0,
                adaptive_calibrate_every: int = 200,
                 use_vision: bool = False,
                 vision_feature_dim: int | None =1024,  # ViT 输出维度，比如 1024/768
                 max_vision_tokens: int = 0,  # 视觉 token 上限
                 vision_mlp_ratio: float = 4.0,
                 vision_use_swiglu: bool = True,

                **kwargs):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id,pad_token_id=pad_token_id,**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers         #transformer层数
        self.num_attention_heads = num_attention_heads     #q head数
        self.num_key_value_heads = num_key_value_heads     #kv head数，q head % k head=0
        self.max_position_embeddings = max_position_embeddings # 理论上下文极限值，已达到tokenizer 上限，扩展需修改tokenizer
        self.RoPE_base=RoPE_base                           # RoPE时频率相关theta base大小，越大支持上下文越多，目前支持最大4k，1e6可支持16k，但针对短文本敏感降低
        self.rope_type=rope_type                        #RoPE时是否开启动态NTK扩展当前base 4k的能力到16k，一般训练可关闭
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
        self.use_ssm=use_ssm
        self.ssm_layers=ssm_layers
        self.embeddingdropout=embeddingdropout          #embedding dropout概率，默认0，最大0.05
        self.learnable_temp= learnable_temp             #是否对qkrms_norm启用一个可学习的增幅
        self.use_affine=use_affine                      #是否对rms_norm启用一个可学习的增幅
        self.use_sampleatt=use_sampleattion             #是否使用自己写的attention，效率低，无cuda优化
        self.use_swiGLU=use_swiGLU                      #MLP是否使用swiGLU，False使用up-silu-down
        self.mlp_ratio=mlp_ratio                        #MLP参数量2*mlp_ratio*H^2
        self.ignore_index=ignore_index
        self.loss_reduction = str(loss_reduction)
        self.sample_mean_alpha = float(sample_mean_alpha)
        if self.loss_reduction not in {"token_mean", "sample_mean", "hybrid"}:
            raise ValueError(
                "loss_reduction must be one of: token_mean, sample_mean, hybrid"
            )
        if not 0.0 <= self.sample_mean_alpha <= 1.0:
            raise ValueError("sample_mean_alpha must be within [0, 1]")
        self.drop_path=drop_path                        #attention MLP 残差分支最大droppath的概率,随着层数线性增长至最大值
        self.residual_dropout=residual_dropout          #attention MLP 残差分支逐元素drop的概率
        self.use_moe=use_moe
        self.moe_layers=moe_layers                      #moe加载在哪些层
        self.num_expert=num_expert
        self.moe_use_detach=moe_use_detach              # false让主损失的梯度回到moe gate，训练不稳但能真的帮助gate学会使用不同的expert，否则gate趋于平均分布
        self.moe_cap_factor=moe_cap_factor              #1-1.5之间选择，决定每个expert加载的最大token量
        self.moe_aux_weight=moe_aux_weight
        self.use_checkpoint = use_checkpoint                    #开启训练慢降显存
        self.checkpoint_use_reentrant = checkpoint_use_reentrant
        self.use_adaptive_softmax=use_adaptive_softmax
        self.adaptive_cutoffs= adaptive_cutoffs
        self.adaptive_div=adaptive_div
        self.adaptive_calibrate_every=adaptive_calibrate_every
        self.use_tcn=use_tcn
        self.tcn_layers=tcn_layers
        self.mlp_ratio = mlp_ratio
        
        self.drop_high_loss=drop_high_loss
        self.mlp_ratio_front = float(mlp_ratio_front)
        self.mlp_ratio_mid   = float(mlp_ratio_mid)
        self.mlp_ratio_back  = float(mlp_ratio_back)
        self.mlp_mid_start   = mlp_mid_start
        self.mlp_back_start  = mlp_back_start
        self.mlp_ratio_overrides = mlp_ratio_overrides or {}
        self.eso_loss_radio=eso_loss_radio
        self.use_vision = use_vision
        self.vision_feature_dim = vision_feature_dim
        self.max_vision_tokens = max_vision_tokens
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_use_swiglu = vision_use_swiglu
        self.vision_dropout = vision_dropout if vision_dropout is not None else dropout
        self.num_query_tokens=num_query_tokens
