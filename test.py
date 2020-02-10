
import torchvision
import torch
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          transform=transform)
testloader = torch.utils.data.DataLoader(cifar_test, batch_size=32, shuffle=True)
# restore entire net1 to net2
net = torch.load('net.pkl')

for name, param in net.named_parameters():
	print(name, param)
# 构造测试的dataloader
#dataiter = iter(testloader)
# 预测正确的数量和总数量
correct = 0
total = 0
# 使用torch.no_grad的话在前向传播中不记录梯度，节省内存
with torch.no_grad():
    for data in testloader:
        images,labels = data
        outputs = net(images)
        # 我们的网络输出的实际上是个概率分布，去最大概率的哪一项作为预测分类
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct +=(predicted==labels).sum().item()
print('Accuracy of the network on the 10000  test images: %d %%' % (100*correct/total))