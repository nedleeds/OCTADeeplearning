import os 
import torch
import torch.cuda
import torch.nn as nn

from torchvision import models
from Clinicadl.cnn_transformer import CNN_Transformer


class ResNet_3D(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_classes = num_class
        self.rgb2gray = self.cnnRGB2GRAY()
        self.resnet_3d = self.preTrained()
        self.fc = self.fc_layers(400, self.num_classes)

    def cnnRGB2GRAY(self):
        conv_layer = nn.Sequential(
                        nn.Conv3d(1, 3, kernel_size=(1,1,1), stride=(16,1,1), padding=(1,1,1), bias=False),
                        nn.BatchNorm3d(3),
                        nn.ReLU(),
                    )
        return conv_layer

    def preTrained(self):
        return torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

    def fc_layers(self, in_c, out_c):
        fc = nn.Sequential(
                nn.Flatten(),
                # nn.Dropout(p=0.5),
                nn.Linear(in_c, 200), # 200
                # nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(200, out_c) # 200
            )
        return fc
        
    def forward(self, x):
        out = self.rgb2gray(x)
        out = self.resnet_3d(out)
        out = self.fc(out)
        return out
    
class ResNet_2D(nn.Module):
    def __init__(self, num_class, ae, depth=152):
        super().__init__()
        self.depth = depth
        self.ae = ae
        self.num_classes = num_class
        self.resnet = self.preTrained()
        self.fc = self.fcLayer(self.num_classes)

    def preTrained(self):
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        os.environ['TORCH_HOME'] = f'./models/res_{self.depth}_2d'
        os.makedirs(f'./models/res_{self.depth}_2d', exist_ok=True) 
    
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        
        model = torch.hub.load('pytorch/vision:v0.10.0', 
                               f'resnet{self.depth}', 
                               pretrained = self.ae)
                               
        conv_weight = model.conv1.weight
        model.conv1.in_channels = 1
        model.conv1.weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))
        self.fc_input = model.fc.out_features
        return model
        
    def fcLayer(self, out_c):
        return nn.Sequential(nn.Linear(self.fc_input, out_c, bias=False))
    
    def forward(self, x):
        out = self.resnet(x)
        out = self.fc(out)
        return out

class VGG_2D(nn.Module):
    def __init__(self, num_class, ae, depth):
        super().__init__()
        super().__init__()
        self.depth = depth
        self.ae = ae
        self.num_classes = num_class
        self.vgg = self.preTrained()
        self.fc = self.fcLayer(self.num_classes)

    def preTrained(self):
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        os.environ['TORCH_HOME'] = f'./models/vgg_{self.depth}_2d'
        os.makedirs(f'./models/vgg_{self.depth}_2d', exist_ok=True) 
    
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

        model = torch.hub.load('pytorch/vision:v0.10.0', 
                                f'vgg{self.depth}_bn', 
                                pretrained=self.ae)

        first_Conv2d_weight = model.features[0].weight
        model.features[0].in_channels = 1
        model.features[0].weight = torch.nn.Parameter(first_Conv2d_weight.sum(dim=1, keepdim=True))
        self.fc_input = model.classifier[-1].out_features
        return model
        
    def fcLayer(self, out_c):
        return nn.Sequential(nn.Linear(self.fc_input, out_c, bias=False))
    
    def forward(self, x):
        out = self.vgg(x)
        out = self.fc(out)
        return out

class GOOGLE_2D(nn.Module):
    def __init__(self, num_class, ae):
        super().__init__()
        self.ae = ae
        self.num_classes = num_class
        self.incept = self.preTrained()
        self.fc = self.fcLayer(self.num_classes)

    def preTrained(self):
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        os.environ['TORCH_HOME'] = f'./models/google_2d'
        os.makedirs(f'./models/google_2d', exist_ok=True) 
    
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

        model = models.googlenet(pretrained = self.ae, progress = True) 
        
        first_conv_weight = model.conv1.conv.weight
        model.conv1.conv.in_channels = 1
        model.conv1.conv.weight = torch.nn.Parameter(first_conv_weight.sum(dim=1, keepdim=True))
        self.fc_input = model.fc.out_features
        return model
        
    def fcLayer(self, out_c):
        return nn.Linear(self.fc_input, out_c, bias=False)
    
    def forward(self, x):
        out = self.incept(x)
        if len(out)>3:
            out = self.fc(out)
        else:
            out = self.fc(out[0])
        return out

class INCEPT_V3_2D(nn.Module):
    def __init__(self, num_class, ae):
        super().__init__()
        self.ae = ae
        self.num_classes = num_class
        self.incept = self.preTrained()
        self.fc = self.fcLayer(self.num_classes)

    def preTrained(self):
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        os.environ['TORCH_HOME'] = f'./models/incept_v3_2d'
        os.makedirs(f'./models/incept_v3_2d', exist_ok=True) 
    
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

        model = models.inception_v3(pretrained = self.ae, progress = True)

        first_conv_weight = model.Conv2d_1a_3x3.conv.weight
        model.Conv2d_1a_3x3.conv.in_channels = 1
        model.Conv2d_1a_3x3.conv.weight = torch.nn.Parameter(first_conv_weight.sum(dim=1, keepdim=True))
        self.fc_input = model.fc.out_features
        return model
        
    def fcLayer(self, out_c):
        return nn.Linear(self.fc_input, out_c, bias=False)
    
    def forward(self, x):
        out = self.incept(x)
        if len(out)>3:
            out = self.fc(out)
        else:
            out = self.fc(out[0])
        return out

class EFFICIENT_2D(nn.Module):
    def __init__(self, num_class, ae):
        super().__init__()
        self.ae = ae
        self.num_classes = num_class
        self.efficient = self.preTrained()
        self.fc = self.fcLayer(self.num_classes)

    def preTrained(self):
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        os.environ['TORCH_HOME'] = f'./models/efficient_2d'
        os.makedirs(f'./models/efficient_2d', exist_ok=True) 
    
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

        model = models.efficientnet_b0(pretrained = self.ae, progress = True)

        first_conv_weight = model.features[0][0].weight
        model.features[0][0].in_channels = 1
        model.features[0][0].weight = torch.nn.Parameter(first_conv_weight.sum(dim=1, keepdim=True))
        self.fc_input = model.classifier[-1].out_features
        return model
        
    def fcLayer(self, out_c):
        return nn.Linear(self.fc_input, out_c, bias=False)
    
    def forward(self, x):
        out = self.efficient(x)
        if len(out)>3:
            out = self.fc(out)
        else:
            out = self.fc(out[0])
        return out

class VIT_2D(nn.Module):
    def __init__(self, num_class, ae):
        super().__init__()
        self.ae = ae
        self.num_classes = num_class
        self.vit = self.ViT()
        self.fc = self.fcLayer(self.num_classes)
        
    def ViT(self):
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        os.environ['TORCH_HOME'] = f'./models/vit_2d'
        os.makedirs(f'./models/vit_2d', exist_ok=True) 
    
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

        model = models.vit_b_32(pretrained = self.ae, progress = False)

        first_conv_weight = model.conv_proj.weight
        model.conv_proj.in_channels = 1
        model.conv_proj.weight = torch.nn.Parameter(first_conv_weight.sum(dim=1, keepdim=True))
        self.fc_input = model.heads.head.out_features
        return model
        
    def fcLayer(self, out_c):
        return nn.Linear(self.fc_input, out_c, bias=False)
    
    def forward(self, x):
        out = self.vit(x)
        return out

class VGG16_2D(nn.Module):
    def __init__(self, num_class, ae):
        super().__init__()
        self.ae = ae
        self.num_classes = num_class
        self.rgb2gray = self.cnnRGB2GRAY(1, 3)
        self.vgg16 = self.preTrained()
        self.fc = self.fcLayer(self.num_classes)

    def cnnRGB2GRAY(self, in_c, out_c):
        conv_layer = nn.Sequential(
                        nn.Conv2d(in_c, out_c, 3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU()
                    )
        return conv_layer

    def preTrained(self):
        if self.ae :
            # return models.vgg16_bn(pretrained=True)
            torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
            os.environ['TORCH_HOME'] = './models/vgg16'
            os.makedirs('./models/vgg16', exist_ok=True) 
            return torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
        else:
            torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
            os.environ['TORCH_HOME'] = './models/vgg16'
            os.makedirs('./models/vgg16', exist_ok=True) 
            return torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=False)

    def fcLayer(self, out_c):
        linear1 = nn.Linear(1000, out_c, bias=False)
        return nn.Sequential(linear1)
        
    def forward(self, x):
        out = self.rgb2gray(x)
        out = self.vgg16(out)
        out = self.fc(out)
        return out

class VGG16_3D(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class  = num_class
        self.convolutions = self.conv_layers()
        self.fc = self.fc_layers(32*8*12*12, self.num_class)
        
    def conv_layers(self):
        conv_layer = nn.Sequential()
        conv_layer.add_module('conv1',self.conv3d_2set(1,2))
        conv_layer.add_module('conv2',self.conv3d_2set(2,4))
        conv_layer.add_module('conv3',self.conv3d_3set(4,8))
        conv_layer.add_module('conv4',self.conv3d_3set(8,16))
        conv_layer.add_module('conv5',self.conv3d_3set(16,32))
        return conv_layer

    def fc_layers(self, in_c, out_c):
        fc_layer = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(in_c, 4096),
                        nn.ReLU(),
                        nn.Linear(4096, 1300),
                        nn.ReLU(),
                        nn.Linear(1300, 50),
                        nn.ReLU(),
                        nn.Linear(50, out_c),
                    )
        return fc_layer

    def conv3d_2set(self, in_c, out_c):
        conv_block = nn.Sequential(
                        nn.Conv3d(in_c, out_c, 3, stride=1, padding=1),
                        nn.InstanceNorm3d(out_c),
                        nn.ReLU(),
                        nn.Conv3d(out_c,out_c, 3, stride=1, padding=1),
                        nn.InstanceNorm3d(out_c),
                        nn.ReLU(),
                        nn.MaxPool3d((2, 2, 2))
                    )
        return conv_block
    
    def conv3d_3set(self, in_c, out_c):
        conv_block = nn.Sequential(
                        nn.Conv3d(in_c,  out_c, 3, stride=1, padding=1),
                        nn.InstanceNorm3d(out_c),
                        nn.ReLU(),
                        nn.Conv3d(out_c, out_c, 3, stride=1, padding=1),
                        nn.InstanceNorm3d(out_c),
                        nn.ReLU(),
                        nn.Conv3d(out_c, out_c, 3, stride=1, padding=1),
                        nn.InstanceNorm3d(out_c),
                        nn.ReLU(),
                        nn.MaxPool3d((2, 2, 2))
                    )
        return conv_block

    def forward(self, x):
        out = self.convolutions(x)
        return self.fc(out)

class CV5FC2_3D(nn.Module): # 13*32 = 416
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes  = num_classes
        self.convolutions = self.conv_layers()
        self.fc           = self.fc_layers(1*8*6*6, self.num_classes) # 1*8*6*6 : Flatten
        
    def conv_layers(self):
        conv_layer = nn.Sequential()                           #  In : 1 * (256 -> 224 -> 416) * 192 * 192
        conv_layer.add_module('conv1', self.conv3d_1set(1, 1, (1, 1, 1))) # out : 1 * (128 -> 112 -> 208) *  96 *  96
        conv_layer.add_module('conv2', self.conv3d_1set(1, 1, (1, 1, 1))) # out : 1 * ( 64 ->  56 -> 104) *  48 *  48
        conv_layer.add_module('conv3', self.conv3d_1set(1, 1, (1, 1, 1))) # out : 1 * ( 16 ->  28 ->  52) *  24 *  24
        conv_layer.add_module('conv4', self.conv3d_1set(1, 1, (1, 1, 1))) # out : 1 * ( 32 ->  14 ->  26) *  12 *  12
        conv_layer.add_module('conv5', self.conv3d_1set(1, 1, (1, 1, 1))) # out : 1 * (  8 ->   7 ->  13) *   6 *   6
        return conv_layer

    def conv3d_1set(self, in_c, out_c, s):
        conv_block = nn.Sequential(
                        nn.Conv3d(in_c, out_c, 3, stride=s, padding=1),
                        nn.BatchNorm3d(out_c),
                        nn.LeakyReLU(),
                        nn.MaxPool3d(2,2)
                    )
        return conv_block

    def fc_layers(self, in_c, out_c):
        fc = nn.Sequential(
                nn.Flatten(),
                # nn.Dropout(p=0.5),
                nn.Linear(in_c, 200), # 200
                # nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(200, out_c) # 200
            )
        return fc

    def forward(self, x):
        out = self.convolutions(x)
        return self.fc(out)


class CV3FC2_3D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes  = num_classes
        self.convolutions = self.conv_layers()
        self.fc           = self.fc_layers(32*50*50, self.num_classes)# 8*8*12*12
        
    def conv_layers(self):
        conv_layer = nn.Sequential()
        conv_layer.add_module('conv1', self.conv3d_1set(1, 1)) # out : 1 * 128 * 200 * 200
        conv_layer.add_module('conv2', self.conv3d_1set(1, 1)) # out : 1 *  64 * 100 * 100
        conv_layer.add_module('conv3', self.conv3d_1set(1, 1)) # out : 1 *  32 *  50 *  50
        return conv_layer

    def conv3d_1set(self, in_c, out_c):
        conv_block = nn.Sequential(
                        nn.Conv3d(in_c, out_c, 3, stride=1, padding=1),
                        nn.BatchNorm3d(out_c),
                        nn.LeakyReLU(),
                        nn.MaxPool3d(2,2)
                    )
        return conv_block

    def fc_layers(self, in_c, out_c):
        fc = nn.Sequential(
                nn.Flatten(),
                # nn.Dropout(p=0.5),
                nn.Linear(in_c, 1000),
                # nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(1000, out_c)
            )
        return fc

    def forward(self, x):
        out = self.convolutions(x)
        return self.fc(out)

class NonLocalBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        super(NonLocalBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = dim_inner  
        self.dim_out = dim_out

        self.theta = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.maxpool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))
        self.phi = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.g = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

        self.out = nn.Conv3d(dim_inner, dim_out, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.bn = nn.BatchNorm3d(dim_out)

    def forward(self, x):
        residual = x

        batch_size = x.shape[0]
        mp = self.maxpool(x)
        theta = self.theta(x)
        phi = self.phi(mp)
        g = self.g(mp)

        theta_shape_5d = theta.shape
        theta, phi, g = theta.view(batch_size, self.dim_inner, -1), phi.view(batch_size, self.dim_inner, -1), g.view(batch_size, self.dim_inner, -1)
      
        theta_phi = torch.bmm(theta.transpose(1, 2), phi) # (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
        theta_phi_sc = theta_phi * (self.dim_inner**-.5)
        p = F.softmax(theta_phi_sc, dim=-1)

        t = torch.bmm(g, p.transpose(1, 2))
        t = t.view(theta_shape_5d)

        out = self.out(t)
        out = self.bn(out)

        out = out + residual
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride, use_nl=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1), stride=(temp_stride, 1, 1), padding=(temp_conv, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        outplanes = planes * 4
        self.nl = NonLocalBlock(outplanes, outplanes, outplanes//2) if use_nl else None


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.nl is not None:
            out = self.nl(out)

        return out

class Res50_3D(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=2, use_nl=False):
        self.inplanes = 1
        super().__init__()
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        nonlocal_mod = 2 if use_nl else 1000
        self.layer1 = self._make_layer(block, self.inplanes*1, layers[0], stride=1, temp_conv=[1, 1, 1],          temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block, self.inplanes*2, layers[1], stride=2, temp_conv=[1, 0, 1, 0],       temp_stride=[1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer3 = self._make_layer(block, self.inplanes*2, layers[2], stride=2, temp_conv=[1, 0, 1, 0, 1, 0], temp_stride=[1, 1, 1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer4 = self._make_layer(block, self.inplanes*4, layers[3], stride=2, temp_conv=[0, 1, 0],          temp_stride=[1, 1, 1])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(4096, num_classes)
        self.drop = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride, nonlocal_mod=1000):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0]!=1:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=(temp_stride[0], stride, stride), padding=(0, 0, 0), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0], False))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i], i%nonlocal_mod==nonlocal_mod-1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.drop(x)

        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        return x
############################################################################################################

class autoencoder(nn.Module):
    def __init__(self, num_class, model=None, device=torch.cuda):
        super(autoencoder, self).__init__()
        self.num_class = num_class
        self.device  = device 
        self.encoder = CNN_Transformer(model).encoder
        self.decoder = CNN_Transformer(model).decoder
        self.total   = nn.Sequential()        
    
    def forward(self, x):
        pool_idx = [None]*len(self.encoder)
        enc_size = []
        total = []
        og_x = x

        for i, seq in enumerate(self.encoder):
            for layer in seq:
                if isinstance(layer, nn.MaxPool3d):
                    enc_size.append(x.shape)
                    x, pool_idx[i] = layer(x) 
                    # 여기서 indices를 뽑아 내므로 summary를 진행하면 2개 요소가 나온다.
                else: x = layer(x)
                # layer.requires_grad_ = True


        for i, seq in enumerate(self.decoder):
            for layer in seq:
                if isinstance(layer, nn.MaxUnpool3d):
                    idx = pool_idx.pop(-1)
                    x = layer(x, idx, output_size=enc_size[len(self.encoder)-i-1])
                else: x = layer(x)
                # layer.requires_grad_ = True
        return x

class freeze(nn.Module):
    def __init__(self, num_class, model=None, device=torch.cuda):
        super(freeze, self).__init__()
        self.num_class = num_class
        self.device  = device 
        self.convolutions = self.getConv(model)
        self.fc = self.getFC(1008, self.num_class)#
    
    def getConv(self, model):
        num_freezed = 1
        freezing_started_layer = 1
        for conv_idx, p in enumerate(model.convolutions): # 0 -> 4 
            for i, parameter in enumerate(p.parameters()):
                if num_freezed != 0:
                    if (freezing_started_layer-1)<= conv_idx <(freezing_started_layer-1)+num_freezed:    # freezing starts from first layer.
                        parameter.requires_grad = False
                    else:
                        parameter.requires_grad = True
                else:
                    parameter.requires_grad = True
        # num_freezed = 1
        # freezing_started_layer = 5
        # for conv_idx, p in enumerate(model.convolutions): # 0 -> 4 
        #     for parameter in p.parameters(): 
        #         if (freezing_started_layer-1)<= conv_idx <= (freezing_started_layer-1)+num_freezed:    # freezing starts from first layer.
        #             parameter.requires_grad = False
        #         else:
        #             parameter.requires_grad = True
        return model.convolutions
    
    def getFC(self, in_c, out_c):
        fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_c, 200),
                nn.ReLU(),
                nn.Linear(200, out_c)
            )
        return fc

    def forward(self, x):
        out = self.convolutions(x)
        return self.fc(out)

########################################### MED3D ################################################

def generate_model(opt):
    from utils import resnet
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
    
    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict() 
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()
    
    # load pretrain
    if opt.phase != 'test' and opt.pretrain_path:
        print ('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
         
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = [] 
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()