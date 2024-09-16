import torchvision.models as models


FOUNDATION_MODELS = {
    'resnet18': models.resnet18,
    'alexnet': models.alexnet,
    'squeezenet': models.squeezenet1_0,
    'vgg16': models.vgg16,
    'densenet': models.densenet161,
    'inception': models.inception_v3,
    'googlenet': models.googlenet,
    'shufflenet': models.shufflenet_v2_x1_0,
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    'mobilenet_v3_small': models.mobilenet_v3_small,
    'resnext50_32x4d': models.resnext50_32x4d,
    'wide_resnet50_2': models.wide_resnet50_2,
    'mnasnet': models.mnasnet1_0,
}
