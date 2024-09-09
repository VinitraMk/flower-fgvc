import torch.optim as optim
import torch
import torch.nn as nn

from models.ots_models import get_model
from common.utils import get_config

class ResNet(nn.Module):
    def __init__(self, num_classes, get_weights = False):
        super().__init__()
        cfg = get_config()
        device = cfg["device"]
        self.model, _ = get_model("resnet18", get_weights)
        self.model = self.model.to(device)
        self.fc = nn.Linear(self.model.fc.in_features, num_classes, bias = True)
        self.fc = self.fc.to(device)
        self.dim = self.model.fc.in_features
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.relu = self.model.relu
        self.maxpool = self.model.maxpool
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

        self.avgpool = self.model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        with torch.no_grad():
            features = torch.flatten(self.avgpool(x.clone().detach()), 1)
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        return x, features



