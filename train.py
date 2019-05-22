import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.autograd import Variable
import torch.optim as optim

from model import WSDDN
from data_pre import myDataSet

Transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

BATCH_SIZE = 2
net_wsddn = WSDDN('VGG11')
criterion = nn.BCELoss() 
optimizer = optim.SGD(net_wsddn.parameters(), lr = 0.001, momentum = 0.9)

trainData = myDataSet('JPEGImages/', 0, Transform)
testData = myDataSet('JPEGImages/' ,1, Transform)
print('trainData', len(trainData))
print('testData', len(testData))
trainData[0][0].shape
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=False)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

for epoch in range(2):
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainLoader):
        images = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()
        #forward + backward + optimizer
        outputs = net_wsddn(images)
        loss = criterion(outputs , labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 2000 == 1999:
            print('[%d , %5d] loss: %.3f' % (epoch + 1 , i + 1 , running_loss / 2000))
            running_loss = 0.0
print('Finished Training')