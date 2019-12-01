import torch
import torchvision
from efficientnet_pytorch import EfficientNet
import pretrainedmodels
from torch import nn
from torchvision import transforms

models = {
    'alexnet': torchvision.models.alexnet,
    'vgg11': torchvision.models.vgg11,
    'vgg11_bn': torchvision.models.vgg11_bn,
    'vgg13': torchvision.models.vgg13,
    'vgg13_bn': torchvision.models.vgg13_bn,
    'vgg16': torchvision.models.vgg16,
    'vgg16_bn': torchvision.models.vgg16_bn,
    'vgg19': torchvision.models.vgg19,
    'vgg19_bn': torchvision.models.vgg19_bn,
    'resnet18': torchvision.models.resnet18,
    'resnet34': torchvision.models.resnet34,
    'resnet50': torchvision.models.resnet50,
    'resnet101': torchvision.models.resnet101,
    'resnet152': torchvision.models.resnet152,
    'squeezenet1_0': torchvision.models.squeezenet1_0,
    'squeezenet1_1': torchvision.models.squeezenet1_1,
    'densenet121': torchvision.models.densenet121,
    'densenet169': torchvision.models.densenet169,
    'densenet161': torchvision.models.densenet161,
    'densenet201': torchvision.models.densenet201,
    'inception_v3': torchvision.models.inception_v3,
    'googlenet': torchvision.models.googlenet,
    'shufflenet_v2_x0_5': torchvision.models.shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': torchvision.models.shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': torchvision.models.shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': torchvision.models.shufflenet_v2_x2_0,
    'mobilenet_v2': torchvision.models.mobilenet_v2,
    'resnext50_32x4d': torchvision.models.resnext50_32x4d,
    'resnext101_32x8d': torchvision.models.resnext101_32x8d,
    'wide_resnet50_2': torchvision.models.wide_resnet50_2,
    'wide_resnet101_2': torchvision.models.wide_resnet101_2,
    'mnasnet0_5': torchvision.models.mnasnet0_5,
    'mnasnet0_75': torchvision.models.mnasnet0_75,
    'mnasnet1_0': torchvision.models.mnasnet1_0,
    'mnasnet1_3': torchvision.models.mnasnet1_3,
}

efficientnet_models = [
    'efficientnet-b0',
    'efficientnet-b1',
    'efficientnet-b2',
    'efficientnet-b3',
]

senet_models=[
    'SENet',
    'senet154',
    'se_resnet50',
    'se_resnet101',
    'se_resnet152',
    'se_resnext50_32x4d', 
    'se_resnext101_32x4d',
]


class PretrainedModel(nn.Module):
    def __init__(self, model: str = 'resnet18'):
        super(PretrainedModel, self).__init__()
        if model in efficientnet_models:
            pretrained = EfficientNet.from_pretrained(model)
        elif model in senet_models:
            pretrained=pretrainedmodels.__dict__[model](num_classes=1000,
                                                        pretrained='imagenet'),
        else:
            pretrained = models[model](pretrained=True)
        self.model = nn.Sequential(
            pretrained,
            nn.ReLU(inplace=True),
            nn.Linear(1000, 100)
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def get_train_transform(model: str):
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # noinspection PyUnresolvedReferences
    @staticmethod
    def get_valid_transform(model: str):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
