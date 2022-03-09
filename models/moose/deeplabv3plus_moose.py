# ADAPTED FROM: https://github.com/VainF/DeepLabV3Plus-Pytorch

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from . import dlv3p_resnet as resnet
from collections import OrderedDict


class SimpleProbe(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, depth=1, stop_gradient=True, residual=False):
        super().__init__()
        self.stop_gradient = stop_gradient
        self.project = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        if residual:
            self.predict = nn.Sequential(
                *([PredictBlock(hidden_size, residual=True)] * depth),
                nn.Conv2d(hidden_size, num_classes, 1)
            )
        else:
            self.predict = nn.Sequential(
                *([nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU()] * depth),
                nn.Conv2d(hidden_size, num_classes, 1)
            )

    def forward(self, low_level_features, intermediate_features, pooling_features):
        ft = torch.cat([low_level_features, intermediate_features, pooling_features], dim=1)
        if self.stop_gradient:
            ft = ft.detach()
        proj = self.project(ft)
        pred = self.predict(proj)
        return pred


class MultiOutputASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(MultiOutputASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.num_convs = len(self.convs)
        # self.pooling = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )


    def forward(self, x):
        intermediate_features = []
        for conv in self.convs:
            intermediate_features.append(conv(x))
        # intermediate
        aspp_features, aspp_pooling = intermediate_features[:-1], intermediate_features[-1]
        # final projection
        res_proj = self.project(torch.cat(intermediate_features, dim=1))
        return aspp_features, aspp_pooling, res_proj


class DeepLabHeadV3PlusMultiOutput(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=None, probes_depth=1, probes_stop_gradient=True):
        super(DeepLabHeadV3PlusMultiOutput, self).__init__()
        if aspp_dilate is None:
            aspp_dilate = [12, 24, 36]
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # print(">>> probes depth =", probes_depth)
        self.aspp = MultiOutputASPP(in_channels, aspp_dilate)
        self.num_probes = self.aspp.num_convs

        self.probes = nn.ModuleList([
            SimpleProbe(
                2 * 256 + 48, 256, num_classes,
                depth=probes_depth, stop_gradient=probes_stop_gradient, residual=False
            ) for _ in range(self.num_probes)
        ])

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature, train_probes_only=False):
        low_level_feature = self.project(feature['low_level'])
        # aspp
        with torch.set_grad_enabled(not train_probes_only):
            aspp_features, pooling_features, main_proj_features = self.aspp(feature["out"])
        # global features
        output_feature = F.interpolate(main_proj_features, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        # intermediate
        intermediate_upsampled = []
        for int_feats in aspp_features:
            intermediate_upsampled.append(
                F.interpolate(int_feats, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False))
        pooling_features = F.interpolate(pooling_features, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        contextual_preds = [probe(low_level_feature, ft, pooling_features) for probe, ft in zip(self.probes, intermediate_upsampled)]

        with torch.set_grad_enabled(not train_probes_only):
            pred = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
        return contextual_preds, pred

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MultiOutputDeepLab(nn.Module):

    def __init__(self, backbone, classifier, train_probes_only=False):
        super().__init__()
        self.train_probes_only = train_probes_only
        self.backbone = backbone
        self.classifier = classifier
        self.num_probes = classifier.num_probes

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        contextual_x, x = self.classifier(features, train_probes_only=self.train_probes_only)
        x = torch.nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        contextual_x = [torch.nn.functional.interpolate(i_x, size=input_shape, mode='bilinear', align_corners=False) for i_x in contextual_x]
        result["out"] = x
        result["contextual_out"] = contextual_x

        return result

    def train(self, mode):
        if self.train_probes_only:
            super().train(False)
            self.classifier.probes.train(mode)
            self.training = mode
        else:
            super().train(mode)
        return self


class AtrousSeparableConvolution(nn.Module):
    """
    Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

#
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module


def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone, probes_depth=1, train_probes_only=False):
    # print(">> OUTPUT_STRIDE", output_stride)
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)

    inplanes = 2048
    low_level_planes = 256

    if name == 'deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3PlusMultiOutput(inplanes, low_level_planes, num_classes, aspp_dilate,
                                                  probes_depth=probes_depth)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = MultiOutputDeepLab(backbone, classifier, train_probes_only=train_probes_only)
    return model


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone, probes_depth=1, train_probes_only=False):
    if backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride,
                             pretrained_backbone=pretrained_backbone,
                             probes_depth=probes_depth,
                             train_probes_only=train_probes_only)
    else:
        raise NotImplementedError
    return model


# Deeplab v3+

def deeplabv3plus_resnet50_moose(num_classes=21, output_stride=8, pretrained_backbone=True, probes_depth=1, train_probes_only=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for moose.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, probes_depth=probes_depth, train_probes_only=train_probes_only)


def deeplabv3plus_resnet101_moose(num_classes=21, output_stride=8, pretrained_backbone=True, probes_depth=1, train_probes_only=False):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for moose.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, probes_depth=probes_depth, train_probes_only=train_probes_only)

