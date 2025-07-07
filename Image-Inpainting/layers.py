import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import (
    Union,
    Tuple,
)



class SpectralConv2d(nn.Module):
   

    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: Union[Tuple[int], int],
                       stride: Union[Tuple[int], int] = 1,
                       padding: Union[Tuple[int], int] = 0,
                       dilation: Union[Tuple[int], int] = 1,
                       groups: int = 1,
                       bias: bool = True,
                       padding_mode: str = 'zeros'):
       

        super(SpectralConv2d, self).__init__()

        self.conv = nn.utils.spectral_norm(nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                            padding_mode=padding_mode))


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.conv(X)



class GatedConv2d(nn.Module):
  
    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: Union[Tuple[int], int],
                       stride: Union[Tuple[int], int] = 1,
                       padding: Union[Tuple[int], int] = 0,
                       dilation: Union[Tuple[int], int] = 1,
                       groups: int = 1,
                       bias: bool = True,
                       padding_mode: str = 'zeros',
                       activation: torch.nn.Module = nn.LeakyReLU(0.2)):
      

        super(GatedConv2d, self).__init__()

        self.conv_gating = nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias,
                                padding_mode=padding_mode)

        self.conv_feature = nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias,
                                padding_mode=padding_mode)

        self.gating_act = nn.Sigmoid()
        self.feature_act = activation
        self.b_norm = nn.BatchNorm2d(out_channels)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        gating = self.conv_gating(X)
        feature = self.conv_feature(X)

        if self.feature_act is None:
            output = feature * self.gating_act(gating)
        else:
            output = self.feature_act(feature) * self.gating_act(gating)

        output = self.b_norm(output)
        return output



class GatedUpConv2d(nn.Module):

    def __init__(self, *args, scale_factor: int = 2, **kwargs):
       

        super(GatedUpConv2d, self).__init__()
        self.conv = GatedConv2d(*args, **kwargs)
        self.scaling_factor = scale_factor


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = F.interpolate(X, scale_factor=self.scaling_factor)
        return self.conv(X)



class SelfAttention(nn.Module):

    def __init__(self, in_channels: int,
                       inter_channels: int = None):
        

        super(SelfAttention, self).__init__()

        if inter_channels is None:
            inter_channels = in_channels // 8

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.gamma = nn.Parameter(torch.zeros(1))

        self.conv_key = nn.Conv2d(in_channels=in_channels,
                                  out_channels=inter_channels,
                                  kernel_size=1)

        self.conv_query = nn.Conv2d(in_channels=in_channels,
                                    out_channels=inter_channels,
                                    kernel_size=1)

        self.conv_value = nn.Conv2d(in_channels=in_channels,
                                    out_channels=inter_channels,
                                    kernel_size=1)

        self.conv_final = nn.Conv2d(in_channels=inter_channels,
                                    out_channels=in_channels,
                                    kernel_size=1)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        
        batch_size = X.shape[0]     
        channel_count = X.shape[1]  
        height = X.shape[2]        
        width = X.shape[3]         

        
        key = self.conv_key(X)
        query = self.conv_query(X)
        value = self.conv_value(X)

        
        key = key.view(batch_size, self.inter_channels, height * width)
        query = query.view(batch_size, self.inter_channels, height * width)
        value = value.view(batch_size, self.inter_channels, height * width)

        
        query = query.permute(0, 2, 1) 

        
        attention = torch.bmm(query, key) 

       
        attention = torch.softmax(attention, dim=1)

        
        att_value = torch.bmm(value, attention)
        att_value = att_value.view(batch_size, self.inter_channels, height, width)

        
        result = self.conv_final(att_value)

       
        result = self.gamma * result + X
        return result


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation='relu',with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
      
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) 
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height)
        energy =  torch.bmm(proj_query,proj_key) 
        attention = self.softmax(energy) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) 

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out ,attention
        else:
            return out
