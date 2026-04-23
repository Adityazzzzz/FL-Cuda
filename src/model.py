# 3 CNNs
#    - EEFNet
#    - DeepConvNet
#    - ShallowConvNet

import torch
import torch.nn as nn
from collections import OrderedDict

class EEGNet(nn.Module):
    def __init__(self, sample_rate, channels=22, F1=8, D=2, F2=16, time=1001, class_num=4, drop_out=0.25, bn_track=True):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.time = time

        # Block 1: Temporal Filter
        self.block_1 = nn.Sequential(OrderedDict([
            ('zeropad', nn.ZeroPad2d((int(sample_rate/4), int(sample_rate/4), 0, 0))), 
            ('conv', nn.Conv2d(1, self.F1, (1, int(sample_rate/2)))), 
            ('bn', nn.BatchNorm2d(self.F1, track_running_stats=bn_track))
        ]))

        # Block 2: Spatial Filter
        self.block_2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.F1, self.D * self.F1, kernel_size=(channels, 1), groups=self.F1)), 
            ('bn', nn.BatchNorm2d(self.D * self.F1, track_running_stats=bn_track)),
            ('elu', nn.ELU()), 
            ('avgpool', nn.AvgPool2d((1, 4))), 
            ('drop', nn.Dropout(drop_out))
        ]))

        # Block 3: Separable Convolution
        self.block_3 = nn.Sequential(OrderedDict([
            ('zeropad', nn.ZeroPad2d((int(sample_rate/16), int(sample_rate/16), 0, 0))),
            ('conv1', nn.Conv2d(self.D * self.F1, self.D * self.F1, kernel_size=(1, int(sample_rate/8)), groups=self.D * self.F1)), 
            ('conv2', nn.Conv2d(self.D * self.F1, self.F2, kernel_size=(1, 1))),
            ('bn', nn.BatchNorm2d(self.F2, track_running_stats=bn_track)),
            ('elu', nn.ELU()),
            ('avgpool', nn.AvgPool2d((1, 8))),
            ('drop', nn.Dropout(drop_out))
        ]))

        # Classifier
        self.fc = nn.Linear((self.F2 * (self.time // 32)), class_num, bias=True)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def MaxNormConstraint(self):
        for n, p in self.block_2.named_parameters():
            if n == 'conv.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=1.0)
        for n, p in self.fc.named_parameters():
            if n == 'weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)

if __name__ == '__main__':
    x = torch.randn(64, 1, 22, 1001)
    model = EEGNet(sample_rate=250)
    print("Model loaded successfully.")
    print(f"Output shape: {model(x).shape}")