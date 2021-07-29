import torch
import torchvision
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, dropout):
        super(ResNet18, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=True)
        self.drop1 = nn.Dropout(dropout)
 
    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        output = self.drop1(output.squeeze())
        return output