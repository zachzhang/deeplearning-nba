import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvNet(torch.nn.Module):
    # @staticmethod
    def ConvBNLayer(self, model, num_in, num_out, kernel):
        i = len(list(model.modules()))
        model.add_module(module=nn.Conv2d(in_channels=num_in, out_channels=num_out, kernel_size=kernel, padding=1),
                         name='conv' + str(i))
        model.add_module(module=nn.BatchNorm2d(num_out), name='bn_' + str(i))
        model.add_module(module=nn.Dropout(.5), name='drop' + str(i))
        model.add_module(module=nn.ReLU(), name='relu' + str(i))

    def __init__(self):
        super(ConvNet, self).__init__()

        ConvBNLayer = self.ConvBNLayer

        conv = torch.nn.Sequential()
        ConvBNLayer(conv, 3, 64, 3)
        # ConvBNLayer(conv,64,64,3)
        conv.add_module(module=nn.MaxPool2d(2), name='max1')  # use stride ?
        ConvBNLayer(conv, 64, 128, 3)
        # ConvBNLayer(conv,128,128,3)
        conv.add_module(module=nn.MaxPool2d(2), name='max2')  # use stride ?
        ConvBNLayer(conv, 128, 256, 3)
        # ConvBNLayer(conv,256,256,3)
        # ConvBNLayer(conv,256,256,3)
        conv.add_module(module=nn.MaxPool2d(2), name='max3')  # use stride ?
        ConvBNLayer(conv, 256, 512, 3)
        # ConvBNLayer(conv,512,512,3)
        # ConvBNLayer(conv,512,512,3)
        conv.add_module(module=nn.MaxPool2d(2), name='max4')  # use stride ?
        ConvBNLayer(conv, 512, 512, 3)
        # ConvBNLayer(conv,512,512,3)
        conv.add_module(module=nn.MaxPool2d(2), name='max5')

        fc = torch.nn.Sequential()
        fc.add_module(module=nn.Linear(512, 512), name='linear')
        fc.add_module(module=nn.BatchNorm1d(512), name='bn_')
        fc.add_module(module=nn.Dropout(.5), name='drop')
        fc.add_module(module=nn.ReLU(), name='dense_relu')
        fc.add_module(module=nn.Linear(512, 1 , bias = False), name='output')
        fc.add_module(module=nn.Sigmoid(), name='sigmoid')

        self.conv = conv
        self.fc = fc

    def forward(self, x):
        h = self.conv(x)
        h = h.view(-1, 512)
        y_hat = self.fc(h)

        return y_hat


class AudioConvNet(torch.nn.Module):
    # @staticmethod
    def ConvBNLayer(self, model, num_in, num_out, kernel):
        i = len(list(model.modules()))
        model.add_module(module=nn.Conv1d(in_channels=num_in, out_channels=num_out, kernel_size=kernel),
                         name='conv' + str(i))
        model.add_module(module=nn.BatchNorm1d(num_out), name='bn_' + str(i))
        model.add_module(module=nn.Dropout(.5), name='drop' + str(i))
        model.add_module(module=nn.ReLU(), name='relu' + str(i))

    def __init__(self):
        super(AudioConvNet, self).__init__()

        ConvBNLayer = self.ConvBNLayer

        conv = torch.nn.Sequential()
        ConvBNLayer(conv, 13, 32, 3)
        ConvBNLayer(conv, 32, 32, 3)
        conv.add_module(module=nn.MaxPool1d(3), name='max1')  # use stride ?
        ConvBNLayer(conv, 32, 64, 3)
        ConvBNLayer(conv, 64, 64, 3)
        conv.add_module(module=nn.MaxPool1d(2), name='max2')  # use stride ?
        ConvBNLayer(conv, 64, 128, 3)
        ConvBNLayer(conv, 128, 128, 3)
        conv.add_module(module=nn.MaxPool1d(2), name='max3')  # use stride ?
        ConvBNLayer(conv, 128, 256, 3)
        ConvBNLayer(conv, 256, 256, 3)
        conv.add_module(module=nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1),
                         name='output')
        conv.add_module(module=nn.Sigmoid(), name='y')


        self.conv = conv

    def forward(self, x):

        y_hat = self.conv(x).squeeze()

        return y_hat


model= AudioConvNet()

print(model.forward(Variable(torch.randn(64,13,100))).size())