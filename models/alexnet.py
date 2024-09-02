import torch.nn as nn
import torch.optim as optim
import torch

from models.ots_models import get_model

class AlexNet(nn.Module):
    def __init__(self, num_classes, get_weights = False):
        super().__init__()
        self.model, _ = get_model("alexnet", get_weights)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes, bias = True)
        self.dim = self.model.classifier[6].in_features
        self.features = self.model.features
        self.avgpool = self.model.avgpool
        self.classifier = self.model.classifier
        self.feature_extractor = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096)
        )
        #self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, bias = True)
        
    def forward(self, x):
        print('in forward pass')
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        print('flattened array')
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = self.avgpool(features)
        x = self.classifier(x)
        return x, features
        