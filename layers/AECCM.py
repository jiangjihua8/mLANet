
import torch  # 导入 PyTorch
from torch import nn  # 从 PyTorch 导入神经网络模块


# 定义 AttentionfusionContext 类，继承自 nn.Module
class AECCM(nn.Module):
    def __init__(self,configs, inplanes, ratio=0.25):
        super(AECCM, self).__init__()

        self.C = 8
        self.d_model = configs.d_model
        self.inplanes = inplanes  # 输入通道数
        self.ratio = ratio  # 缩减比例
        self.planes = int(inplanes * ratio)  # 缩减后的通道数
        # 定义注意力池化的卷积层和 softmax
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)  # 在维度 2 上做 softmax

        self.channel_add_conv1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),  # 1x1 卷积
            nn.LayerNorm([self.planes, 1, 1]),  # 层归一化
            nn.ReLU(inplace=True),  # 激活函数 ReLU
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1)  # 1x1 卷积，恢复到原通道数
        )

        self.channel_add_conv2 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=3,padding=1),  # 3x3 卷积
            nn.LayerNorm([self.planes, 1, 1]),  # 层归一化
            nn.ReLU(inplace=True),  # 激活函数 ReLU
            nn.Conv2d(self.planes, self.inplanes, kernel_size=3,padding=1)  # 3x3 卷积，恢复到原通道数
        )
        self.conv1 = nn.Conv2d(inplanes,inplanes,1)


    # 定义空间池化方法
    def spatial_attention(self, x):
        batch, channel, N, L = x.size()  # 获取输入张量的形状

        input_x = x.view(batch, channel, N * L)  # 展平 N 和 L
        input_x = input_x.unsqueeze(1)  # 增加一个维度
        context_mask = self.conv_mask(x)  # 应用 1x1 卷积层
        context_mask = context_mask.view(batch, 1, N * L)  # 展平 N 和 L
        context_mask = self.softmax(context_mask)  # 在 N * L 上应用 softmax
        context_mask = context_mask.unsqueeze(-1)  # 增加一个维度
        context = torch.matmul(input_x, context_mask)  # 计算加权和
        context = context.view(batch, channel, 1, 1)  # 恢复形状

        return context

    # 定义前向传播方法
    def forward(self, x):
        _,N,_ = x.shape
        x = x.reshape(-1,self.C,N,self.d_model//self.C)
        spatial_out = self.spatial_attention(x)  # 获取上下文信息
        out = x  # 初始化输出
        conv2_out = self.channel_add_conv2(spatial_out)
        conv1_out = self.channel_add_conv1(spatial_out)
        conv_out = self.conv1(conv2_out+conv1_out)
        out = out + conv_out  # 输出加上融合结果
        out = out.reshape(-1,N,self.d_model)

        return out  # 返回最终输出


# 测试代码块
if __name__ == "__main__":
    in_tensor = torch.ones((1, 7, 720//2, 2))  # 创建一个全1的输入张量，形状为 (1, 64, 128, 128)
    cb = AECCM(inplanes=7, ratio=0.25)  # 创建 ContextBlock 实例
    out_tensor = cb(in_tensor)  # 传递输入张量进行前向传播
    print(in_tensor.shape)  # 打印输入张量的形状
    print(out_tensor.shape)  # 打印输出张量的形状
