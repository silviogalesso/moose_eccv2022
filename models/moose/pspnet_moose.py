import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter


class SegmentationModelMultiOutput(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, train_probes_only=False):
        super(SegmentationModelMultiOutput, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.num_probes = self.classifier.num_probes
        self.train_probes_only = train_probes_only
        self.train()

    def forward(self, x):
        input_shape = x.shape[-2:]
        with torch.set_grad_enabled(not self.train_probes_only):
            features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        intermediate_x, x = self.classifier(x)
        x = torch.nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        intermediate_x = [torch.nn.functional.interpolate(i_x, size=input_shape, mode='bilinear', align_corners=False)
                          for i_x in intermediate_x]
        result["out"] = x
        result["contextual_out"] = intermediate_x
        return result

    def train(self, mode=True):
        if self.train_probes_only:
            # if mode==True: set everything to eval, except for the probes
            # if mode==False: set everything to eval anyway
            super().train(False)
            self.classifier.probes.train(mode)
            self.training = mode
        else:
            super().train(mode)
        return self


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * 512, 512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        return x


class SimpleProbe(nn.Sequential):
    def __init__(self, input_size, hidden_size, output_size, depth):
        assert depth > 0
        super().__init__(nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1, bias=False),
                         nn.BatchNorm2d(hidden_size),
                         nn.ReLU(inplace=True),
                         nn.Dropout2d(0.1),
                         *([nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Dropout2d(0.1)] * (depth - 1)),
                         nn.Conv2d(hidden_size, output_size, 1))


class SimpleProbeProjection(nn.Sequential):
    def __init__(self, input_size, hidden_size, output_size, depth):
        assert depth > 0
        super().__init__(nn.Conv2d(input_size, hidden_size, kernel_size=1, bias=False),
                         nn.BatchNorm2d(hidden_size),
                         nn.ReLU(inplace=True),
                         nn.Dropout2d(0.1),
                         *([nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Dropout2d(0.1)] * (depth - 1)),
                         nn.Conv2d(hidden_size, output_size, 1))


# pyramid pooling
class PPMMultiOutput(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6), probes_depth=1, probes_hidden_size=512,
                 train_probes_only=False, projection_probe=True):
        super(PPMMultiOutput, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * 512, 512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

        self.train_probes_only = train_probes_only
        self.num_probes = len(self.ppm)
        probe = SimpleProbeProjection if projection_probe else SimpleProbe
        self.probes = nn.ModuleList([
            probe(input_size=fc_dim + 512, hidden_size=probes_hidden_size, output_size=num_class, depth=probes_depth)
            for _ in self.ppm
        ])

    def forward(self, conv_out):
        conv5 = conv_out
        input_size = conv5.size()
        ppm_out = []
        int_preds = []
        for pool_scale, int_head in zip(self.ppm, self.probes):
            with torch.set_grad_enabled(not self.train_probes_only):
                o = nn.functional.interpolate(pool_scale(conv5),
                                              (input_size[2], input_size[3]),
                                              mode='bilinear', align_corners=False)
            ppm_out.append(o)
            int_preds.append(int_head(torch.cat([conv5, o], 1).detach()))

        ppm_out = torch.cat([conv5] + ppm_out, 1)

        x = self.conv_last(ppm_out)

        return int_preds, x


def pspnet_resnet101_moose(num_classes, pretrained_backbone=True, probes_depth=1, avg_pooling=False,
                                  probes_hidden_size=512, train_probes_only=False, projection_probes=True):
    backbone = resnet.__dict__["resnet101"](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    inplanes = 2048
    pp_module = PPMMultiOutput
    classifier = pp_module(num_class=num_classes, fc_dim=inplanes,
                           probes_depth=probes_depth, probes_hidden_size=probes_hidden_size,
                           train_probes_only=train_probes_only, projection_probe=projection_probes)

    model = SegmentationModelMultiOutput(backbone, classifier, train_probes_only=train_probes_only)
    return model


def pspnet_resnet50_moose(num_classes, pretrained_backbone=True, probes_depth=1, avg_pooling=False,
                                 probes_hidden_size=512, train_probes_only=False, projection_probes=True):
    backbone = resnet.__dict__["resnet50"](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    inplanes = 2048
    pp_module = PPMMultiOutput
    classifier = pp_module(num_class=num_classes, fc_dim=inplanes,
                           probes_depth=probes_depth, probes_hidden_size=probes_hidden_size,
                           train_probes_only=train_probes_only, projection_probe=projection_probes)

    model = SegmentationModelMultiOutput(backbone, classifier, train_probes_only=train_probes_only)
    return model
