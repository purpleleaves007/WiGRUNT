import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo
import torchvision.models as models

class Resnet34(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet34, self).__init__()

        resnet = models.resnet34(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2],
        #                              nn.AdaptiveAvgPool2d(1)
        #                              ) # before avgpool

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1

        # fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512
        #
        self.fc = nn.Linear(512, 7)  # new fc layer 512x7
        # self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet50, self).__init__()

        resnet = models.resnet50(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2],
        #                              nn.AdaptiveAvgPool2d(1)
        #                              ) # before avgpool

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1

        # fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512
        #
        self.fc = nn.Linear(2048, 7)  # new fc layer 512x7
        # self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Resnet101(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet101, self).__init__()

        resnet = models.resnet101(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2],
        #                              nn.AdaptiveAvgPool2d(1)
        #                              ) # before avgpool

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1

        # fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512
        #
        self.fc = nn.Linear(2048, 7)  # new fc layer 512x7
        # self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Resnet181(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet181, self).__init__()

        resnet = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2],
        #                              nn.AdaptiveAvgPool2d(1)
        #                              ) # before avgpool

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1

        # fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512
        #
        self.fc = nn.Linear(512, 112)  # new fc layer 512x7
        # self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Resnet152(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet152, self).__init__()

        resnet = models.resnet152(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2],
        #                              nn.AdaptiveAvgPool2d(1)
        #                              ) # before avgpool

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1

        # fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512
        #
        self.fc = nn.Linear(2048, 7)  # new fc layer 512x7
        # self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    # inplanes其实就是channel,叫法不同
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 把shortcut那的channel的维度统一
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=7):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        #downsample 主要用来处理H(x)=F(x)+x中F(x)和xchannel维度不匹配问题
        downsample = None
        #self.inplanes为上个box_block的输出channel,planes为当前box_block块的输入channel
        #在shotcut中若维度或者feature_size不一致则需要downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        #只在这里传递了stride=2的参数，因而一个box_block中的图片大小只在第一次除以2
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #[2, 2, 2, 2]和结构图[]X2是对应的
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained: #加载模型权重
        pretrained_resnet18_params = torch.load(r'D:\人脸识别预训练模型\resnet18\ijba_res18_naive.pth.tar')
        # model_dict = model.state_dict()
        # new_state_dict = {k: v for k, v in pretrained_resnet18_params.items() if k in model_dict}
        # model_dict.update(new_state_dict)
        # model.load_state_dict(model_dict)

        pretrained_resnet18_params['state_dict'].pop('module.fc.weight')
        pretrained_resnet18_params['state_dict'].pop('module.fc.bias')
        model.load_state_dict(pretrained_resnet18_params, strict=False) #加载预训练模型参数除了最后一层

    return model

"""
VGG模型
"""
class VGG(nn.Module):

    def __init__(self, features, num_classes=6, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model

class Resnet18c(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet18c, self).__init__()

        resnet = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2],
        #                              nn.AdaptiveAvgPool2d(1)
        #                              ) # before avgpool

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1
        # fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512
        #
        self.fc1 = nn.Linear(512, 6)
        #self.fc2 = nn.Linear(256, 6)  # new fc layer 512x7
        # self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #x = self.fc2(x)
        return x

class Resnet50c(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet50c, self).__init__()

        resnet = models.resnet50(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2],
        #                              nn.AdaptiveAvgPool2d(1)
        #                              ) # before avgpool

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1

        # fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512
        #
        self.fc = nn.Linear(2048, 6)  # new fc layer 512x7
        # self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x