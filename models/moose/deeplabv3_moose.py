import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3, ASPP, ASPPConv, ASPPPooling
from collections import OrderedDict


class DeepLabHeadCustom(nn.Sequential):
    def __init__(self, in_channels, num_classes, atrous_rates=None):
        if atrous_rates is None:
            atrous_rates = [12, 24, 36]
        super(DeepLabHeadCustom, self).__init__(
            ASPP(in_channels, atrous_rates=atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class MultiOutputASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        self.num_convs = len(modules)

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        intermediate_features = []
        for conv in self.convs:
            intermediate_features.append(conv(x))
        aspp_features, aspp_pooling = intermediate_features[:-1], intermediate_features[-1]
        return aspp_features, aspp_pooling, self.project(torch.cat(intermediate_features, dim=1))


class PredictBlock(nn.Module):
    def __init__(self, hidden_size, residual=False):
        super().__init__()
        self.conv = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_size)
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        if self.residual:
            out += identity
        out = self.relu(out)
        return out


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

    def forward(self, intermediate_features, pooling_features):
        ft = torch.cat([intermediate_features, pooling_features], dim=1)
        if self.stop_gradient:
            ft = ft.detach()
        proj = self.project(ft)
        pred = self.predict(proj)
        return pred


class MultiOutputDeepLabHead(nn.ModuleList):
    # Horrible hack of nn.ModuleList
    def __init__(self, in_channels, num_classes, atrous_rates=None, probes_stop_gradient=True, probes_depth=1, residual_probes=False):
        if atrous_rates is None:
            atrous_rates = [12, 24, 36]
        super().__init__([
            MultiOutputASPP(in_channels, atrous_rates=atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        ])

        # self.num_probes = len(self._modules['0'].convs)
        self.num_probes = self._modules['0'].num_convs

        self.probes = nn.ModuleList([
            SimpleProbe(
                2*256, 256, num_classes,
                depth=probes_depth, stop_gradient=probes_stop_gradient, residual=residual_probes
            ) for _ in range(self.num_probes)
        ])

    def forward(self, x, train_probes_only=False):
        with torch.set_grad_enabled(not train_probes_only):
            aspp_features, pooling_features, main_proj_features = self._modules['0'](x)
        contextual_preds = [probe(ft, pooling_features) for probe, ft in zip(self.probes, aspp_features)]
        with torch.set_grad_enabled(not train_probes_only):
            pred = self._modules['1'](main_proj_features)
            pred = self._modules['2'](pred)
            pred = self._modules['3'](pred)
            pred = self._modules['4'](pred)
        return contextual_preds, pred


class MultiOutputDeepLab(nn.Module):
    def __init__(self, backbone, classifier, train_probes_only=False):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.num_probes = classifier.num_probes
        self.train_probes_only = train_probes_only

    def forward(self, x):
        input_shape = x.shape[-2:]

        with torch.set_grad_enabled(not self.train_probes_only):
            features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        contextual_x, x = self.classifier(x, train_probes_only=self.train_probes_only)
        x = torch.nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        contextual_x = [torch.nn.functional.interpolate(i_x, size=input_shape, mode='bilinear', align_corners=False) for i_x in contextual_x]
        result["out"] = x
        result["contextual_out"] = contextual_x

        return result

    def train(self, mode):
        if self.train_probes_only:
            # if mode==True: set everything to eval, except for the probes
            # if mode==False: set everything to eval anyway
            super().train(False)
            self.classifier.probes.train(mode)
            self.training = mode
        else:
            super().train(mode)
        return self


def _deeplabv3_resnet_multi_output(name, backbone_name, num_classes, atrous_rates, pretrained_backbone=True,
                                   probes_stop_gradient=True, probes_depth=1, train_probes_only=False, residual_probes=False):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    inplanes = 2048
    classifier = MultiOutputDeepLabHead(inplanes, num_classes, atrous_rates, probes_stop_gradient=probes_stop_gradient,
                                        probes_depth=probes_depth, residual_probes=residual_probes)
    base_model = MultiOutputDeepLab

    model = base_model(backbone, classifier, train_probes_only=train_probes_only)
    return model


def deeplabv3_resnet50_moose(num_classes=21, atrous_rates=None, pretrained_backbone=True,
                             probes_stop_gradient=True, probes_depth=1,
                             train_probes_only=False, residual_probes=False):
    model = _deeplabv3_resnet_multi_output("deeplabv3", "resnet50", num_classes, atrous_rates,
                                           pretrained_backbone=pretrained_backbone,
                                           probes_stop_gradient=probes_stop_gradient,
                                           probes_depth=probes_depth,
                                           train_probes_only=train_probes_only,
                                           residual_probes=residual_probes)
    return model


def deeplabv3_resnet101_moose(num_classes=21, atrous_rates=None, pretrained_backbone=True,
                              probes_stop_gradient=True, intermediate_heads_depth=1,
                              train_probes_only=False, residual_probes=False):
    model = _deeplabv3_resnet_multi_output("deeplabv3", "resnet101", num_classes, atrous_rates,
                                           pretrained_backbone=pretrained_backbone,
                                           probes_stop_gradient=probes_stop_gradient,
                                           probes_depth=intermediate_heads_depth,
                                           train_probes_only=train_probes_only)
    return model


