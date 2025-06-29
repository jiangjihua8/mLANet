import torch
from torch import nn

class AECCM(nn.Module):
    def __init__(self, configs, inplanes, ratio=0.25):
        super(AECCM, self).__init__()
        self.d_model = configs.d_model
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
        )
        self.channel_add_conv2 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=3, padding=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=3, padding=1)
        )
        self.conv1 = nn.Conv2d(inplanes, inplanes, 1)

    def spatial_attention(self, x):
        batch, channel, N, L = x.size()
        input_x = x.view(batch, channel, N * L)
        input_x = input_x.unsqueeze(1)
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(batch, 1, N * L)
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.unsqueeze(-1)
        context = torch.matmul(input_x, context_mask)
        context = context.view(batch, channel, 1, 1)
        return context

    def forward(self, x):
        _, N, _ = x.shape
        x = x.reshape(-1, self.inplanes, N, self.d_model//self.inplanes)
        spatial_out = self.spatial_attention(x)
        out = x
        conv2_out = self.channel_add_conv2(spatial_out)
        conv1_out = self.channel_add_conv1(spatial_out)
        conv_out = self.conv1(conv2_out+conv1_out)
        out = out + conv_out
        out = out.reshape(-1, N, self.d_model)
        return out

if __name__ == "__main__":
    in_tensor = torch.ones((1, 7, 720//2, 2))
    cb = AECCM(inplanes=7, ratio=0.25)
    out_tensor = cb(in_tensor)
    print(in_tensor.shape)
    print(out_tensor.shape)
