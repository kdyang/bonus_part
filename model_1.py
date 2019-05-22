import torch
import torchvision.models as v_models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.autograd import Variable

from spp_layer import spatial_pyramid_pool
from data_pre import myDataSet

BATCH_SIZE = 2
R = 20

kdy = torch.Tensor(BATCH_SIZE, R, 512, 14, 14)
#print(kdy.shape)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class WSDDN(nn.Module):
    def __init__(self, vgg_name):
        super(WSDDN, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8c = nn.Linear(4096, 20)
        self.fc8d = nn.Linear(4096, 20)

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        #out = out.view(out.size(0), -1)
        x = spatial_pyramid_pool(previous_conv = x, num_sample = BATCH_SIZE, 
                                previous_conv_size = [x.size(2),x.size(3)], out_pool_size = [2, 2])
        print(x.shape)
        #x = F.relu(self.fc6(x))
        #x = F.relu(self.fc7(x))
        #x_c = F.relu(self.fc8c(x))
        #x_d = F.relu(self.fc8d(x))
        #print(x_c)
        #print(x_d)
        #segma_c = F.softmax(x_c, dim = 1)
        #segma_d = F.softmax(x_d, dim = 0)
        #print(segma_c)
        #print(segma_d)
        #print(segma_c.shape)
        #print(segma_d.shape)
        #x = segma_c * segma_d
        #x = torch.sum(x, dim = 0)
        #print(x.shape)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

for i in range(BATCH_SIZE):
    test_k = spatial_pyramid_pool(previous_conv = kdy[i,:], num_sample = R, 
                                    previous_conv_size = [kdy.size(3),kdy.size(4)], out_pool_size = [2, 2])

print(test_k.shape)
#pretrained_model_path = 
#net_wsddn = WSDDN('VGG11')
#state_dict = torch.load(pretrained_model_path)
#net_wsddn.load_state_dict({k: v for k, v in state_dict.items() if k in net_wsddn.state_dict()})