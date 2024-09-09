from models.alexnet import AlexNet
from models.resnet import ResNet

def get_model(num_classes, model_name = 'alexnet'):
    model = {}
    if model_name == "alexnet":
        model = AlexNet(num_classes, True)
    elif model_name == "resnet18":
        model = ResNet(num_classes, True)
    else:
        raise SystemExit("Error: no valid model name passed! Check run.yaml")
    return model