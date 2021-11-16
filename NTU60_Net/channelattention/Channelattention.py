import torch.nn as nn
import math
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#B*C*H*W->B*C*1*1.....->B*C*1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # print(in_planes,in_planes // 2)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # print('x:',x.shape)
        # print('self.avg_pool(x):',self.avg_pool(x).shape)
        # print('elf.fc1(self.avg_pool(x)):',self.fc1(self.avg_pool(x)).shape)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # print('avg_out:',avg_out.shape)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class ChannelAttention0(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention0, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # print(in_planes,in_planes // 2)
        self.fc1   = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # print('x:',x.shape)
        # print('self.avg_pool(x):',self.avg_pool(x).shape)
        # print('elf.fc1(self.avg_pool(x)):',self.fc1(self.avg_pool(x)).shape)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # print('avg_out:',avg_out.shape)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)