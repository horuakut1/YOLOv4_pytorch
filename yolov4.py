import torch
import torch.nn as nn
from torchsummary import summary
# torch version = 1.2.0
##########################
### CspDarkNet_53 part ###
##########################
class Mish(nn.Module):
    'mish'
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x*torch.tanh(nn.Softplus()(x))

class baseConv(nn.Module):
    'conv_bn_mish'
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,mish=True) -> None:
        super().__init__()
        if mish:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
                nn.BatchNorm2d(out_channels),
                Mish(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self, x):
        return self.conv(x)

class resBlock(nn.Module):
    'residual block'
    def __init__(self,in_channels,out_channels=None) -> None:
        super().__init__()
        if out_channels==None:
            out_channels = in_channels
        self.conv = nn.Sequential(
            baseConv(in_channels,out_channels,kernel_size=1,stride=1,padding=0),
            baseConv(out_channels,in_channels,kernel_size=3,stride=1,padding=1,mish=False),
        )
    def forward(self, x):
        return Mish()(x + self.conv(x))

class cspResBlock(nn.Module):
    'csp residual block'
    def __init__(self,in_channels,out_channels,num_blocks,first=False) -> None:
        super().__init__()
        if first:
            self.downsample = baseConv(in_channels,out_channels,kernel_size=3,stride=2,padding=1)
            self.route1 = baseConv(out_channels,out_channels,kernel_size=1,stride=1,padding=0)
            self.route2 = nn.Sequential(
                baseConv(out_channels,out_channels,kernel_size=1,stride=1,padding=0),
                resBlock(out_channels,out_channels//2),
            )
            self.con = baseConv(int(2*out_channels),out_channels,kernel_size=1,stride=1,padding=0)
        else:
            self.downsample = baseConv(in_channels,out_channels,kernel_size=3,stride=2,padding=1)
            self.route1 = baseConv(out_channels,out_channels//2,kernel_size=1,stride=1,padding=0)
            self.route2 = nn.Sequential(
                baseConv(out_channels,out_channels//2,kernel_size=1,stride=1,padding=0),
                nn.Sequential(*[resBlock(out_channels//2,out_channels//2) for _ in range(num_blocks)])
            )
            self.con = baseConv(out_channels,out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        x = self.downsample(x)
        route1 = self.route1(x)
        route2 = self.route2(x)
        x = torch.cat([route1,route2],dim=1)
        x = self.con(x)
        return x

class CspDarkNet_53(nn.Module):
    'CspDarkNet53 network'
    def __init__(self,num_blocks,num_classes = 10) -> None:
        super().__init__()
        channels = [64,128,256,512,1024]
        self.conv1 = baseConv(3,32,kernel_size=3,stride=1,padding=1)
        self.neck = cspResBlock(32,channels[0],num_blocks=None,first=True)
        self.out1 = nn.Sequential(
            cspResBlock(channels[0],channels[1],num_blocks[0]),
            cspResBlock(channels[1],channels[2],num_blocks[1]),
        )
        self.out2 = cspResBlock(channels[2],channels[3],num_blocks[2])
        self.out3 = cspResBlock(channels[3],channels[4],num_blocks[3])



    def forward(self, x):
        x = self.conv1(x)
        x = self.neck(x)
        out1 = self.out1(x)
        out2 = self.out2(out1)
        out3 = self.out3(out2)
        return out1,out2,out3


###################
### yolov4 part ###
###################
class SPP(nn.Module):
    'Spatial Pyramid Pooling'
    def __init__(self) -> None:
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9,stride=1,padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13,stride=1,padding=6)
    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        return torch.cat([x,x1,x2,x3],dim=1)

class baseConvleaky(nn.Module):
    'conv_bn_leaky'
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return self.conv(x)

class con_x3(nn.Module):
    'Perform 3 convolution operations around SPP, out_channels[0] is the real out_channels, the channels can be changed through this class'
    def __init__(self,in_channels,out_channels=[0,0]) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            baseConvleaky(in_channels,out_channels[0],kernel_size=1,stride=1,padding=0),
            baseConvleaky(out_channels[0],out_channels[1],kernel_size=3,stride=1,padding=1),
            baseConvleaky(out_channels[1],out_channels[0],kernel_size=1,stride=1,padding=0),
        )
    def forward(self, x):
        return self.conv(x)

class con_x5(nn.Module):
    'Perform 5 convolution operations in PANet, out_channels[0] is the real out_channels, the channels can be changed through this class'
    def __init__(self,in_channels,out_channels=[0,0]) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            baseConvleaky(in_channels,out_channels[0],kernel_size=1,stride=1,padding=0),
            baseConvleaky(out_channels[0],out_channels[1],kernel_size=3,stride=1,padding=1),
            baseConvleaky(out_channels[1],out_channels[0],kernel_size=1,stride=1,padding=0),
            baseConvleaky(out_channels[0],out_channels[1],kernel_size=3,stride=1,padding=1),
            baseConvleaky(out_channels[1],out_channels[0],kernel_size=1,stride=1,padding=0),
        )
    def forward(self, x):
        return self.conv(x)

class upsample(nn.Module):
    'conv + upsample in PANet'
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            baseConvleaky(in_channels,out_channels,kernel_size=1,stride=1,padding=0),
            nn.Upsample(scale_factor=2,mode='nearest')
        )
    def forward(self, x):
        return self.conv(x)

class yolohead(nn.Module):
    'yolo head, out_channels[1] is the real out_channels'
    def __init__(self,in_channels,out_channels=[0,0]) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            baseConvleaky(in_channels,out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.Conv2d(out_channels[0],out_channels[1],kernel_size=1),
        )
    def forward(self, x):
        return self.conv(x)

class yolov4_net(nn.Module):
    'the main yolov4 network'
    def __init__(self,num_anchors,num_classes) -> None:
        super().__init__()
        self.backbone = CspDarkNet_53([2,8,8,4])

        # out3 part
        self.conx3_1 = con_x3(1024,[512,1024])  # channel from 1024 to 512
        self.spp = SPP()                        # spp out channel = 4 * channel, there is 512*4 = 2048
        self.conx3_2 = con_x3(2048,[512,1024])  # channel from 2048 to 512
        self.upsample_1 = upsample(512,256)     # channel from 512 to 256, which will cat with out2
        self.conx5_1 = con_x5(1024,[512,1024])  # has cated with out2 downsample, so channel is from 1024 to 512
        self.yolohead_1 = yolohead(512,[1024,int(num_anchors*(5+num_classes))]) # channel from 512 to ..., if num_anchors = 3 and num_classes =20, ... will be 75

        # out2 part
        self.conv_1 = baseConvleaky(512,256,kernel_size=1,stride=1,padding=0)   # channel from 512 to 256
        self.conx5_2 = con_x5(512,[256,512])        # has cated with out3 upsample, so channel is from 512 to 256
        self.upsample_2 = upsample(256,128)         # channel from 256 to 128, which will cat with out1
        self.conx5_3 = con_x5(512,[256,512])        # has cated with out1 downsample, so channel is from 512 to 256
        self.downsample_1 = baseConvleaky(256,512,kernel_size=3,stride=2,padding=1) # channel from 256 to 512, which will cat with out3
        self.yolohead_2 = yolohead(256,[512,int(num_anchors*(5+num_classes))]) # channel from 256 to ..., if num_anchors = 3 and num_classes =20, ... will be 75

        # out1 part
        self.conv_2 = baseConvleaky(256,128,kernel_size=1,stride=1,padding=0)       # channel from 256 to 128
        self.conx5_4 = con_x5(256,[128,256])                                        # has cated with out2 upsample, so channel is from 256 to 128
        self.downsample_2 = baseConvleaky(128,256,kernel_size=3,stride=2,padding=1) # channel from 128 to 256, which will cat with out2
        self.yolohead_3 = yolohead(128,[256,int(num_anchors*(5+num_classes))])      # channel from 128 to ..., if num_anchors = 3 and num_classes =20, ... will be 75


    def forward(self, x):
        # This part is the PANet structure
        out1,out2,out3 = self.backbone(x)

        out3 = self.conx3_1(out3)
        out3 = self.spp(out3)
        out3 = self.conx3_2(out3)
        out3_up = self.upsample_1(out3)

        out2 = self.conv_1(out2)
        out2 = torch.cat([out3_up,out2],dim=1)
        out2 = self.conx5_2(out2)
        out2_up = self.upsample_2(out2)

        out1 = self.conv_2(out1)
        out1 = torch.cat([out2_up,out1],dim=1)
        out1 = self.conx5_4(out1)
        out1_down =self.downsample_2(out1)
        out1 = self.yolohead_3(out1)

        out2 = torch.cat([out1_down,out2],dim=1)
        out2 = self.conx5_3(out2)
        out2_down = self.downsample_1(out2)
        out2 = self.yolohead_2(out2)

        out3 = torch.cat([out2_down,out3],dim=1)
        out3 = self.conx5_1(out3)
        out3 = self.yolohead_1(out3)

        return out3,out2,out1


if __name__ == '__main__':
    model = yolov4_net(3,20)
    device = torch.device('cuda')
    model.to(device)
    summary(model=model,input_size=(3,416,416))

