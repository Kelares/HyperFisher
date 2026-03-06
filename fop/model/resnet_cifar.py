"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import numpy as np
import cv2
import logging
import time

def get_model(cfg, num_classes):
    model = Network(cfg, mode="train", num_classes=num_classes)

    model = model.cuda()

    return model


class GAP(nn.Module):
    """Global Average pooling
        Widely used in ResNet, Inception, DenseNet, etc.
     """

    def __init__(self):
        super(GAP, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avgpool(x)
        #         x = x.view(x.shape[0], -1)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# for LDAM Loss
class FCNorm(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FCNorm, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.linear(F.normalize(x), F.normalize(self.weight))
        return out


class LWS(nn.Module):

    def __init__(self, num_features, num_classes, bias=True):
        super(LWS, self).__init__()
        self.fc = nn.Linear(num_features, num_classes, bias=bias)
        self.scales = nn.Parameter(torch.ones(num_classes))
        for param_name, param in self.fc.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        x *= self.scales
        return x


class Network(nn.Module):
    def __init__(self, cfg, mode="train", num_classes=1000):
        super(Network, self).__init__()
        pretrain = (
            True
            if mode == "train"
            and cfg.RESUME_MODEL == ""
            and cfg.BACKBONE.PRETRAINED_MODEL != ""
            else False
        )

        self.num_classes = num_classes
        self.cfg = cfg

        self.backbone = eval(self.cfg.BACKBONE.TYPE)(
            self.cfg,
            pretrain=pretrain,
            pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL,
            last_layer_stride=2,
        )
        self.mode = mode
        self.module = self._get_module()
        self.classifier = self._get_classifer()

        if cfg.NETWORK.PRETRAINED and os.path.isfile(cfg.NETWORK.PRETRAINED_MODEL):
            try:
                self.load_model(cfg.NETWORK.PRETRAINED_MODEL)
            except:
                raise ValueError('network pretrained model error')

    def forward(self, x, **kwargs):
        if "feature_flag" in kwargs or "feature_cb" in kwargs or "feature_rb" in kwargs:
            return self.extract_feature(x, **kwargs)
        elif "classifier_flag" in kwargs:
            return self.classifier(x)
        elif 'feature_maps_flag' in kwargs:
            return self.extract_feature_maps(x)
        elif 'layer' in kwargs and 'index' in kwargs:
            if kwargs['layer'] in ['layer1', 'layer2', 'layer3']:
                x = self.backbone.forward(x, index=kwargs['index'], layer=kwargs['layer'], coef=kwargs['coef'])
            else:
                x = self.backbone(x)
            x = self.module(x)
            if kwargs['layer'] == 'pool':
                x = kwargs['coef']*x+(1-kwargs['coef'])*x[kwargs['index']]
            x = x.view(x.shape[0], -1)
            x = self.classifier(x)
            if kwargs['layer'] == 'fc':
                x = kwargs['coef']*x + (1-kwargs['coef'])*x[kwargs['index']]
            return x

        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def get_backbone_layer_info(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            layers = 3
            blocks_info = [5, 5, 5]
        elif 'res10' in self.cfg.BACKBONE.TYPE:
            layers = 4
            blocks_info = [1, 1, 1, 1]
        else:
            layers = 4
            blocks_info = [3, 4, 6, 3]
        return layers, blocks_info

    def extract_feature(self, x, **kwargs):
        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)
        return x

    def extract_feature_maps(self, x):
        x = self.backbone(x)
        return x

    def freeze_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False


    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        print("Backbone model has been loaded...")


    def load_model(self, model_path, tau_norm=False, tau=1):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if  k == 'classifier.weight' and tau_norm:
                print('*-*'*30)
                print('Using tau-normalization')
                print('*-*'*30)
                v = v / torch.pow(torch.norm(v, 2, 1, keepdim=True), tau)
            new_dict[k] = v

        if self.mode == 'train' and self.cfg.CLASSIFIER.TYPE == "cRT":
            print('*-*'*30)
            print('Using cRT')
            print('*-*'*30)
            for k in new_dict.keys():
                if 'classifier' in k: print(k)
            new_dict.pop('classifier.weight')
            try:
                new_dict.pop('classifier.bias')
            except:
                pass

        if self.mode=='train' and self.cfg.CLASSIFIER.TYPE == "LWS":
            print('*-*'*30)
            print('Using LWS')
            print('*-*'*30)
            bias_flag = self.cfg.CLASSIFIER.BIAS
            for k in new_dict.keys():
                if 'classifier' in k: print(k)
            class_weight = new_dict.pop('classifier.weight')
            new_dict['classifier.fc.weight'] = class_weight
            if bias_flag:
                class_bias = new_dict.pop('classifier.bias')
                new_dict['classifier.fc.bias'] = class_bias

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        if self.mode == 'train' and self.cfg.CLASSIFIER.TYPE in ['cRT', 'LWS']:
            self.freeze_backbone()
        print("All model has been loaded...")


    def get_feature_length(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            num_features = 64
        elif 'res10' in self.cfg.BACKBONE.TYPE:
            num_features = 512
        else:
            num_features = 2048
        return num_features


    def _get_module(self):
        module_type = self.cfg.MODULE.TYPE
        if module_type == "GAP":
            module = GAP()
        elif module_type == "Identity":
            module= Identity()
        else:
            raise NotImplementedError

        return module


    def _get_classifer(self):
        bias_flag = self.cfg.CLASSIFIER.BIAS
        num_features = self.get_feature_length()
        if self.cfg.CLASSIFIER.TYPE == "FCNorm":
            classifier = FCNorm(num_features, self.num_classes)
        elif self.cfg.CLASSIFIER.TYPE in ["FC", "cRT"]:
            classifier = nn.Linear(num_features, self.num_classes, bias=bias_flag)
        elif self.cfg.CLASSIFIER.TYPE == "LWS":
            classifier = LWS(num_features, self.num_classes, bias=bias_flag)
        else:
            raise NotImplementedError

        return classifier


    def cam_params_reset(self):
        self.classifier_weights = np.squeeze(list(self.classifier.parameters())[0].detach().cpu().numpy())

    def get_CAM_with_groundtruth(self, image_idxs, dataset, label_list, size):
        ret_cam = []
        size_upsample = size
        for i in range(len(image_idxs)):
            idx = image_idxs[i]
            label = label_list[idx]
            self.eval()
            with torch.no_grad():
                img = dataset._get_trans_image(idx)
                feature_conv = self.forward(img.to('cuda'), feature_maps_flag=True).detach().cpu().numpy()
            b, c, h, w = feature_conv.shape
            assert b == 1
            feature_conv = feature_conv.reshape(c, h*w)
            cam = self.classifier_weights[label].dot(feature_conv)
            del img
            del feature_conv
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255*cam_img)
            ret_cam.append(cv2.resize(cam_img, size_upsample))
        return ret_cam



def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Cifar(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet_Cifar, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "last_linear" not in k and "classifier" not in k and "linear" not in k and "fd" not in k:
                k = k.replace("backbone.", "")
                k = k.replace("fr", "layer3.4")
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def forward(self, x, **kwargs):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer1':
            out = kwargs['coef']*out + (1-kwargs['coef'])*out[kwargs['index']]
        out = self.layer2(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer2':
            out = kwargs['coef']*out+(1-kwargs['coef'])*out[kwargs['index']]
        out = self.layer3(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer3':
            out = kwargs['coef']*out+(1-kwargs['coef'])*out[kwargs['index']]
        return out

def res32_cifar(
    cfg=None,
    pretrain=True,
    pretrained_model="",
    last_layer_stride=2,
):
    resnet = ResNet_Cifar(BasicBlock, [5, 5, 5])
    if pretrain and pretrained_model != "":
        resnet.load_model(pretrain=pretrained_model)
    else:
        print("Choose to train from scratch")
    return resnet