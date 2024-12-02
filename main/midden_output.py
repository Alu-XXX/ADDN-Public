import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


def set_midden_input(idx:int, cell, weights, input):
    global midden_input1,midden_input2,midden_input3
    global net
    net=BottleneckBlock(*input.shape[:2]).cuda()
    if idx==1:
        #midden_input1.reshape(input.size())
        midden_input1=input
    elif idx==2:
        #midden_input2.reshape(input.size())
        midden_input2=input
    elif idx==3:
        #midden_input3.reshape(input.size())
        midden_input3=input
    else:
        return -1
    
def get_midden_output(idx:int):
    if idx==1:
        return midden_output1
    elif idx==2:
        return midden_output2
    elif idx==3:
        return midden_output3
    else:
        return -1    
    
def set_midden_fea(idx:int, input):
    global midden_fea1,midden_fea2,midden_fea3
    if idx==1:
        midden_output1=input
        midden_fea1=net(input)
    elif idx==2:
        midden_output2=input
        midden_fea2=net(input)
    elif idx==3:
        midden_output3=input
        midden_fea3=net(input)
    else:
        return -1
    
def get_midden_fea(idx:int):
    if idx==1:
        return midden_fea1
    elif idx==2:
        return midden_fea2
    elif idx==3:
        return midden_fea3
    else:
        return -1

def set_midden_output(idx:int, input,num_class):
    #net_op=midden_op(input.shape[1],num_class).cuda()
    net_op=midden_op().cuda()
    global midden_output1,midden_output2,midden_output3
    if idx==1:
        midden_fea1=input
        output=net(input)
        midden_output1=net_op(output)
    elif idx==2:
        midden_fea2=input
        output=net(input)
        midden_output2=net_op(output)
    elif idx==3:
        midden_fea3=input
        output=net(input)
        midden_output3=net_op(output)
    else:
        return -1
    
class midden_op(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net=nn.Sequential(
            nn.AvgPool2d(kernel_size=1,stride=1),
            nn.Softmax()
        )
    
    def forward(self, x):
        return self.net(x)

class BottleneckBlock(nn.Module):
    expansion = 1 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes,kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = torch.nn.Conv2d(planes, planes*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        output = output.expand(residual.shape)


        output_f=output.clone()

        output_f += residual
        
        output_f = self.relu(output_f)

        return output_f

def branchBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out//4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )


