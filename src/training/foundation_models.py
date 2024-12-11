from torchvision import models


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self._convert_recursively(self)

    def _convert_recursively(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = AttrDict(v)


FOUNDATION_MODELS = AttrDict(
    {
        'tiny': {
            'squeezenet': models.squeezenet1_0,  # 1.2M
            'shufflenet': models.shufflenet_v2_x1_0,  # 2.3M
            'mobilenet_v3_small': models.mobilenet_v3_small,  # 2.5M
            'mobilenet_v2': models.mobilenet_v2,  # 3.5M
            'mnasnet': models.mnasnet1_0,  # 4.4M
            'efficientnet_b0': models.efficientnet_b0,  # 5.3M
            'mobilenet_v3_large': models.mobilenet_v3_large,  # 5.5M
        },
        'small': {
            'efficientnet_b1': models.efficientnet_b1,  # 7.8M
            'densenet121': models.densenet121,  # 8.0M
            'efficientnet_b2': models.efficientnet_b2,  # 9.1M
            'resnet18': models.resnet18,  # 11.7M
            'efficientnet_b3': models.efficientnet_b3,  # 12.2M
            'googlenet': models.googlenet,  # 13.0M
            'densenet169': models.densenet169,  # 14.1M
        },
        'medium': {
            'efficientnet_b4': models.efficientnet_b4,  # 19.3M
            'densenet201': models.densenet201,  # 20.0M
            'resnet34': models.resnet34,  # 21.8M
            'resnext50_32x4d': models.resnext50_32x4d,  # 25.0M
            'resnet50': models.resnet50,  # 25.6M
            'inception': models.inception_v3,  # 27.2M
            'densenet161': models.densenet161,  # 28.7M
            'efficientnet_b5': models.efficientnet_b5,  # 30.4M
        },
        'large': {
            'efficientnet_b6': models.efficientnet_b6,  # 43.0M
            'resnet101': models.resnet101,  # 44.5M
            'resnet152': models.resnet152,  # 60.2M
            'alexnet': models.alexnet,  # 61.1M
            'efficientnet_b7': models.efficientnet_b7,  # 66.3M
            'wide_resnet50_2': models.wide_resnet50_2,  # 68.9
        },
        'huge': {
            'vgg11': models.vgg11,  # 132.9M
            'vgg11_bn': models.vgg11_bn,  # 132.9M
            'vgg16': models.vgg16,  # 138.4M
            'vgg19': models.vgg19,  # 143.7M
            'vgg19_bn': models.vgg19_bn,  # 143.7M
        }
    }
)
