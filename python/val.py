import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torchvision
import scipy.io as io
import numpy as np
from data_loader import datatecsival
from model1 import *
from plot import show_acc_curv
from options import Options
from da_att import SpatialGate, ChannelGate

class DACN(nn.Module):
    def __init__(self, num_classes=10):
        super(DACN, self).__init__()
        self.spa = SpatialGate()
        self.cga = ChannelGate(512)
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512,8)
        
    def forward(self, x):       
        outspa, afspa = self.spa(x)
        out = self.features(outspa) 
        outcga, afcga = self.cga(out)
        out = self.avgpool(outcga)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out,outspa,outcga,afspa,afcga


def run(device):
    args = Options().initialize()

    global_train_acc = []
    global_test_acc = []

    img_transform = transforms.Compose([
        transforms.Resize([args.img_size, args.img_size]),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    test_list = r'H:/Widar3.0/QFM/STIFMM'

    dataset_target = datatecsival(
        data_list=test_list,
        transform=img_transform
    )

    testloader = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=2)

    pthfile = r'./model/bestmodel.pth'
    net = torch.load(pthfile)
    net =  net.to(device)

    criterion = nn.CrossEntropyLoss()

    params = filter(lambda p: p.requires_grad, net.parameters())
    print("Waiting Test!")
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs,outpspa,outpcga,afespa,afecga = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            afespa = np.array(outpcga.cpu())
            io.savemat('save.mat',{'outspa':afespa})
        print('Test Acc：%.3f%%' % (100. * correct / total))
        acc = 100. * correct / total


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = False
    print(device)
    run(device)
