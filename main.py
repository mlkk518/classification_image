import torch

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from  model import  LeNet
import torch.optim as optim

from tensorboardX import SummaryWriter





# cifar-10官方提供的数据集是用numpy array存储的
# 下面这个transform会把numpy array变成torch tensor，然后把rgb值归一到[0, 1]这个区间
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 在构建数据集的时候指定transform，就会应用我们定义好的transform
# root是存储数据的文件夹，download=True指定如果数据不存在先下载数据
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)


print(cifar_train)


trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=32, shuffle=True)


net = LeNet()

criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

print("Start Traing ...")
j=0
writer = SummaryWriter('runs')
for epoch in range(80):
    # 我们用一个变量来记录每100个batch的平均loss

    loss100 = 0.0
    for i, data in enumerate(trainloader):
        inputs,labels=data
        optimizer.zero_grad()
        outputs=net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        loss100 +=loss.item()
        if i%100 ==99:
            loss_mean_100=loss100/100
            print('[epoch %d,Batch %5d] loss: %.3f' %(epoch + 1,i+1,loss_mean_100))
            loss100 =0.0
            j=j+1
            writer.add_scalar('loss_mean_100', loss_mean_100, global_step=j)

writer.close()
# 2 ways to save the net
torch.save(net, 'net.pkl')  # save entire net
#torch.save(net.state_dict(), 'net_params.pkl')  # save only the parameters

print("Done Training !")






