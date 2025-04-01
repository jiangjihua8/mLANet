import torch
import torch.nn as nn
from einops import einops
# from torch.utils.tensorboard import SummaryWriter



from layers.PyramidalRNNEmbedding import PyramidalRNNEmbedding

from layers.mlstm.mlstm import mLSTMLayer,mLSTMLayerConfig


from layers.AECCM  import AECCM



class RevIN(nn.Module):
    def __init__(self, num_variates, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.num_variates = num_variates
        self.gamma = nn.Parameter(torch.ones(num_variates, 1))
        self.beta = nn.Parameter(torch.zeros(num_variates, 1))

    def forward(self, x):
        # x~b t d
        x = einops.rearrange(x, 'b n v -> b v n')
        assert x.shape[1] == self.num_variates

        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        var_rsqrt = var.clamp(min = self.eps).rsqrt() # rsqrt()?????????????????????
        instance_normalized = (x - mean) * var_rsqrt
        rescaled = instance_normalized * self.gamma + self.beta

        rescaled = einops.rearrange(rescaled, 'b v n -> b n v')

        def reverse_fn(scaled_output):
            scaled_output = einops.rearrange(scaled_output, 'b n v -> b v n')
            clamped_gamma = torch.sign(self.gamma) * self.gamma.abs().clamp(min = self.eps)
            unscaled_output = (scaled_output - self.beta) / clamped_gamma
            scaled_output = unscaled_output * var.sqrt() + mean
            scaled_output = einops.rearrange(scaled_output, 'b v n -> b n v')
            return scaled_output

        return rescaled, reverse_fn

class Model(nn.Module):


    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # self.output_attention = configs.output_attention
        # PRE Embedding
        self.enc_embedding = PyramidalRNNEmbedding(configs.convWindows, configs.d_model, configs.rnnMixTemperature, configs.dropout)

        self.reversible_instance_norm = RevIN(configs.enc_in)

        self.mlstm = mLSTMLayer(mLSTMLayerConfig(
            embedding_dim=configs.d_model,  # 输入嵌入的维度
            conv1d_kernel_size=configs.conv1d_kernel_size,  # 一维卷积核的大小
            qkv_proj_blocksize=configs.qkv_proj_blocksize,  # QKV投影的块大小
            num_heads=configs.num_heads,  # mLSTM的头数
            proj_factor=configs.proj_factor,  # 上投影的比例因子
            bias=configs.bias,  # 是否使用偏置
            dropout=0.4,  # Dropout 概率
            context_length=configs.context_length  # 序列长度
        ))

        self.aeccm  = AECCM(configs,inplanes=8, ratio=0.5)

        self.dropout = nn.Dropout(0.4)

        self.layernorm = nn.LayerNorm(configs.d_model)
        self.sigmoid = nn.Sigmoid()
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc):
        # RevIN
        x_enc, reverse_fn = self.reversible_instance_norm(x_enc)

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates


        # Embedding
        # B L N -> B N E
        x = self.enc_embedding(x_enc) # covariates (e.g timestamp) can be also embedded as tokens

        a_out = self.aeccm(x)

        m_out = self.mlstm(x)*a_out +x
        l_out = self.layernorm(m_out)
        out =self.dropout( l_out+ self.mlstm(m_out) + x)


        # B N E -> B N S -> B S N
        e_out = self.projector(out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        e_out = reverse_fn(e_out)

        return e_out

    def forward(self, x_enc):
        e_out = self.forecast(x_enc)
        return e_out[:, -self.pred_len:, :]  # [B, L, D]
