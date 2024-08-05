import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torchvision
from data_loader import datatrcsi,datatecsi,datatrcsip,datatecsip,datatrcsio,datatecsio,datatrcsie,datatecsie
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
        self.fc = nn.Linear(512,6)
        
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
    train_list = r'H:/Widar3.0/QFM/STIFMM'
 
    dataset_source = datatrcsie(
        data_list=train_list,
        transform=img_transform
    )

    trainloader = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=2)

    test_list = r'H:/Widar3.0/QFM/STIFMM'

    dataset_target = datatecsie(
        data_list=test_list,
        transform=img_transform
    )

    testloader = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=2)

    # model = args.model
    # net = model()
    net =  DACN().to(device)
    #net = Resnet101(pretrained=True).to(device)
    #net = Resnet152(pretrained=True).to(device)
    #net = Resnet50(pretrained=True).to(device)
    #net = Resnet34(pretrained=True).to(device)
    # for name, value in net.named_parameters():
    #     if 'fc' not in name:
    #         value.requires_grad = False


    writer = SummaryWriter('./logs_%s' % args.model)
    criterion = nn.CrossEntropyLoss()

    params = filter(lambda p: p.requires_grad, net.parameters())
    #optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    best_acc = 50
    print("Start Training, %s!" % args.model)
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(0, args.Epoch):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):

                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs,outpspa,outpcga,afespa,afecga = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()
                    writer.add_scalar('train_loss', sum_loss / (i + 1), epoch + 1)
                    global_train_acc.append(100. * correct / total)

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
                    print('Test Acc：%.3f%%' % (100. * correct / total))
                    acc = 100. * correct / total
                    if len(global_test_acc) != 0:
                        if acc > max(global_test_acc):
                            print('Saving model......')
                            torch.save(net, '%s/bestmodel.pth' % ('./model/'))

                    global_test_acc.append(acc)
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    writer.add_scalar('test_acc', acc, epoch + 1)
                    if acc > best_acc: 
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
                scheduler.step()
            show_acc_curv(length, global_train_acc, global_test_acc)
            print("Training Finished, TotalEPOCH=%d" % args.Epoch)

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = False
    print(device)
    run(device)
