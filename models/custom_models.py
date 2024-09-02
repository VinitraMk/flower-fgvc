from models.alexnet import AlexNet

def get_model(num_classes, model_name = 'vit'):
    model = {}
    if model_name == "alexnet":
        model = AlexNet(num_classes, True)
    else:
        raise SystemExit("Error: no valid model name passed! Check run.yaml")
    return model