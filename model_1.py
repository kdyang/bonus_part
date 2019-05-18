import torch
import torchvision.models as v_models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.autograd import Variable

#from torch.utils import data
#from PIL import Image
from spp_layer import spatial_pyramid_pool
from data_pre import myDataSet

BATCH_SIZE = 1
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
        self.fc6 = nn.Linear(9216, 9216)
        self.fc7 = nn.Linear(9216, 9216)
        self.fc8c = nn.Linear(9216, 9216)
        self.fc8d = nn.Linear(9216, 9216)

    def forward(self, x):
        x = self.features(x)
        #out = out.view(out.size(0), -1)
        x = spatial_pyramid_pool(previous_conv = x, num_sample = BATCH_SIZE, 
                                previous_conv_size = [int(x.size(2)),int(x.size(3))], out_pool_size = [4, 1, 1])
        
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

net_wsddn = WSDDN('VGG11')

Transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

trainData = myDataSet('JPEGImages/', 0, Transform)
testData = myDataSet('JPEGImages/' ,1, Transform)
print('trainData', len(trainData))
print('testData', len(testData))
trainData[0][0].shape
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=False)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

for i, (images, labels) in enumerate(trainLoader):
        images = Variable(images)
        labels = Variable(labels)
        out_test = net_wsddn(images)
        print(images.shape)
        print(out_test.shape)

#pretrained_model_path = 
#net_wsddn = WSDDN('VGG11')
#state_dict = torch.load(pretrained_model_path)
#net_wsddn.load_state_dict({k: v for k, v in state_dict.items() if k in net_wsddn.state_dict()})