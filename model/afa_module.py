import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import clip


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes) 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class FirstOctaveConv(nn.Module):   # 对应第一个红色和绿色
      
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]   # 3
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * in_channels), # (512,256)
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, in_channels - int(alpha * in_channels), 
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):    # x：n,c,h,w
        if self.stride ==2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x) # 低频
        X_h = x
        X_h = self.h2h(X_h)   # 高频
        X_l = self.h2l(X_h2l) # 低频

        return X_h, X_l

class OctaveConv(nn.Module): # 低、高频输入，低、高频输出 对应第二个红色和绿色
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        # 低到低，通道缩一半
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        
        # 低到高，改变输出通道
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        
        # 高到低，输出通道减一半，改变输入通道
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        
        # 高到高，输出、输入通道都改变
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]),int(X_h2h.size()[3])), mode='bilinear')

        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l

class LastOctaveConv(nn.Module): # 低频和高频对齐输出
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()   # 继承 nn.Module 的一些属性和方法
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * out_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(out_channels - int(alpha * out_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2h = self.h2h(X_h) # 高频组对齐通道
        X_l2h = self.l2h(X_l) # 低频组对齐通道
        # 低频组对齐长宽尺寸
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear') 

        X_h = X_h2h + X_l2h  
        return X_h       

class Octave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(Octave, self).__init__()
        # 第一层，将特征分为高频和低频
        self.fir = FirstOctaveConv(in_channels, out_channels, kernel_size)

        # 第二层，低高频输入，低高频输出
        self.mid1 = OctaveConv(in_channels, in_channels, kernel_size)   # 同频输入、输出
        self.mid2 = OctaveConv(in_channels, out_channels, kernel_size)  # 不同频输入、输出

        # 第三层，将低高频汇合后输出
        self.lst = LastOctaveConv(in_channels, out_channels, kernel_size)

    def forward(self, x):   
        x0 = x
        x_h, x_l = self.fir(x)                   
        x_hh, x_ll = x_h, x_l,
        # x_1 = x_hh +x_ll
        x_h_1, x_l_1 = self.mid1((x_h, x_l))     
        x_h_2, x_l_2 = self.mid1((x_h_1, x_l_1)) 
        x_h_5, x_l_5 = self.mid2((x_h_2, x_l_2)) 
        x_ret = self.lst((x_h_5, x_l_5)) 
        return x_ret
    
class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel=64):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 放大两倍


        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 128, 1)  

    def forward(self, x1, x2, x3):  # for (low, mid, hig)
        
        x_low = x1  
        x_mid = x2  
        x_hig = x3  
        
        x_2_1 = x_low * self.conv_upsample1(self.upsample(input=x_mid))   
        x_2_2 = x_mid * self.conv_upsample1(F.interpolate(x_hig, (25,25), mode='bilinear', align_corners=True))   # ([4, 64, 25, 25])
        
        x_3_1 = x_2_1 * self.conv_upsample1(self.upsample(x_2_2))   
        
        c_3_2 = torch.cat((x_2_2, self.conv_upsample1(F.interpolate(x_hig, (25,25), mode='bilinear', align_corners=True))), 1)

        c_4 = torch.cat((x_3_1, self.conv_upsample5(self.upsample(input=c_3_2))), 1) 
        c_4 = self.conv_concat3(c_4)   
        
        x = self.conv4(c_4)
        x = self.conv5(x)

        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]#format with class
            texts = clip.tokenize(texts).to(device) # tokenize
            class_embeddings = model.encode_text(texts) # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights.t()



