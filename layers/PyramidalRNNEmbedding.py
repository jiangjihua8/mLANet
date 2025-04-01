
import numpy as np

def splitWindowList(input_list):
    result_lists = []
    gcd_list = []
    input_label = np.zeros(len(input_list))
    input_length = len(input_list)
    # labeledCount = 0 #统计所有被分配的总数 这样不好 有bug
    while input_label.sum() <input_length:
        tmpList = None
        lastValue = None
        for i in range(input_length-1,-1,-1):
            label = input_label[i]
            value = input_list[i]
            if label == 0 or i == 0  : #or ( (lastValue is not None) and (lastValue % value == 0)): # 这个条件目的是多个窗口尽量共享，去掉后 只有第一个窗口可以复用多次
                if tmpList is None:
                    tmpList = [value]
                    input_label[i] = 1
                    lastValue = value
                elif lastValue %value ==0:
                    tmpList.append(value)
                    input_label[i] = 1
                    lastValue = value

        if (tmpList is not None) and len(tmpList)!=0:
            tmpList.sort()
            result_lists.append(tmpList)
    return result_lists

def countResultWindows(result_lists):
    cnt = 0
    for windows in result_lists:
        cnt+=len(windows)
    return cnt








import torch
import torch.nn as nn
import torch.nn.functional as F

'''一维卷积块'''
class Conv1dBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super(Conv1dBlock, self).__init__()
        self.conv1d0 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1d0(x)
        return x
class RNNBlock(nn.Module):
    '''输入为Conv1dBlock卷积后的结果'''
    def __init__(self,sequenceDim,hidR):
        super(RNNBlock, self).__init__()
        self.gru = nn.GRU(sequenceDim,hidR,batch_first=True) # 默认gru第一个维度是时间步 batch_first设置第一个是batch
    def forward(self, x):
        x = x.permute(0, 2, 1)  # 变为 批量*通道维度，时间长，子序列窗口(conv的输出通道数目) 方便后续
        _,hn = self.gru(x)
        return hn
class ConvRNNBlock(nn.Module):
    '''输入为Conv1dBlock卷积后的结果'''
    def __init__(self,in_channels,out_channels,kernel_size,stride,hidR):
        super(ConvRNNBlock, self).__init__()
        self.conv1d = Conv1dBlock(in_channels, out_channels, kernel_size,stride)
        self.gru = RNNBlock(out_channels,hidR);
        self.in_channels = in_channels
        self.out_channels = out_channels
    def forward_conv(self,x):
        conv_output = self.conv1d(x)
        return conv_output
    def forward_rnn(self,x):
        rnn_output = self.gru(x)
        return rnn_output

    def forward(self, x):
        conv_output = self.conv1d(x)
        rnn_output = self.gru(conv_output)
        return conv_output,rnn_output

    def getOutChannels(self):
        return self.out_channels
    def getInChannels(self):
        return self.in_channels


import torch.nn.functional as F
'''卷积链之间不共享卷积层,集成由粗至细的'''
class PyramidalRNNEmbedding(nn.Module):
    def __init__(self, windows, d_model,rnnMixTemperature, dropout=0.1):
        super(PyramidalRNNEmbedding, self).__init__()
        # 构建多个卷积链
        split_windows_lists = splitWindowList(windows) #返回划分的倍数列表和最大公约数，每一个倍数有序链表构造一条卷积 # 24 48 72 96 144
        self.split_windows_lists = split_windows_lists # 储存供后边使用
        window_count = countResultWindows(split_windows_lists)
        self.rateParameter = nn.Parameter(torch.ones((window_count,1))/window_count)
        self.temperature = rnnMixTemperature # 通过命令行统一设置
        # 子序列编码
        self.hidRDict = {}
        for window in windows:
            self.hidRDict[window] = 0
        # 对每个窗口的卷积块进行计数
        for sub_windows in split_windows_lists:
            for window in sub_windows:
                self.hidRDict[window]+=1
        # 分配每个卷积窗口的隐层个数
        for window in windows:
            tmpHidR = d_model//len(windows)
            self.hidRDict[window] = int(tmpHidR/self.hidRDict[window]) # 同样的窗口 多个评分hidR 让不同窗口的hidR平均


        moduleLists  = []
        conv_channels = 128 #统一的卷积channel
        for window_list in split_windows_lists:
            in_channels = 1
            convRNNBlocks = []
            for i in range(len(window_list)):
                out_channels = window_list[i] # window 每一层卷积的输出通道数等于窗口大小
                # out_channels = 24 # 设置统一的通道数 方便反卷积
                kernel = int(out_channels/in_channels) # kernel等于与前一个窗口的倍数
                stride = kernel
                hidR = self.hidRDict[out_channels] # window
                if in_channels==1:
                    tmpConvRNNBlock = ConvRNNBlock(1, conv_channels, kernel, stride,hidR)
                else:
                    tmpConvRNNBlock = ConvRNNBlock(conv_channels, conv_channels, kernel, stride,hidR)
                convRNNBlocks.append(tmpConvRNNBlock)
                in_channels = out_channels
            convRnns = nn.ModuleList(convRNNBlocks)
            moduleLists.append(convRnns)
        self.nnModuleLists = nn.ModuleList(moduleLists)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        if len(windows)==5 and d_model!=880:
            self.outLinear = nn.Linear(510, d_model)
        else:
            self.outLinear = nn.Linear(d_model,d_model)
    def forward(self, x):
        # x: [Batch Variate Time]
        batchSize = x.shape[0]
        timeLength = x.shape[1]
        channel = x.shape[2]
        x = x.permute(0, 2, 1).reshape(batchSize * channel, 1, timeLength)

        # if x_mark is None:
        #     x = x.permute(0, 2, 1).reshape(batchSize * channel, 1, timeLength)
        # else:
        #     x = torch.cat([x.permute(0, 2, 1), x_mark.permute(0, 2, 1)],dim=1)
        #     channel+=x_mark.shape[2]
        #     x = x.reshape(batchSize * channel, 1, timeLength)

        hns = [] # 保存所有卷积rnn链的结果
        #对每个卷积链进行操作，得到rnn表示
        for i,moduleList in enumerate(self.nnModuleLists):
            convMap = {} #临时存储卷积结果
            tmpX = x
            window_list = self.split_windows_lists[i]
            for  j in range(len(moduleList)):
                tmpConvRNNBlock = moduleList.__getitem__(j)
                tmpX = tmpConvRNNBlock.forward_conv(tmpX)
                # window = tmpConvRNNBlock.getOutChannels() #这样获取不到窗口大小
                window = window_list[j]
                convMap[window] = tmpX
            # 上采样融合
            lastWindow = None
            upScaleConv = tmpX
            for j in range(len(window_list)-1,-1,-1):
                tmpConvRNNBlock = moduleList.__getitem__(j)
                window = window_list[j]
                if lastWindow is not None: #从第二个开始采样
                    scale_factor = lastWindow/window
                    upScaleConv = nn.functional.interpolate(upScaleConv,scale_factor=scale_factor)# ,mode='bilinear'不能直接用于三维
                    cha = upScaleConv.shape[-1] - convMap[window].shape[-1]
                    if cha !=0: #对于奇数等情况 补上上采样确实的维度
                        upScaleConv = torch.cat([upScaleConv,upScaleConv[:, :, -1:]],dim=-1)
                    rnn_input = upScaleConv + convMap[window]
                    del convMap[window] #释放内存
                else:
                    rnn_input = upScaleConv
                tmpHn = tmpConvRNNBlock.forward_rnn(rnn_input)
                hidR = self.hidRDict[window]  # window 获取window对应的隐层数量
                tmpHn = tmpHn.reshape(batchSize, channel, hidR)
                hns.append(tmpHn)

                lastWindow = window
        rate = torch.softmax(self.rateParameter/self.temperature,-1)  #
        # hns = [(hni*rate[i]) for i, hni in enumerate(list(hns.values()))]
        hns = [(hni*rate[i]) for i, hni in enumerate(hns)]
        hns = torch.cat(hns,dim=-1)
        # print(hns.shape)
        hns = self.outLinear(hns) #类似多头注意力的线性层
        # hns = hns.permute(0, 2, 1)
        return hns

if __name__ == '__main__':
    # Testing the Adaptive_Spectral_Block
    x = torch.randn(512, 660, 321)  # Batch of 2, Sequence length of 3, Feature size of 32
    # x_mark = torch.randn(256, 720, 4)
    asb = PyramidalRNNEmbedding(windows=[22 ,44, 66, 88 ,132], d_model=512,rnnMixTemperature=0.002)
    y = asb(x)
    print(y.shape)
    # for i in range(len(y)):
    #     print(y[i].shape)
    # print(y[1])# Expected output: torch.Size([2, 3, 32])