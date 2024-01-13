import torch
import torch.nn as nn

class MPI_Net(nn.Module):
    def __init__(self, input_channels, num_outputs, ngf=64,Kaiming = True):
        super(MPI_Net, self).__init__()

        self.conv1_1 = nn.Conv2d(input_channels, ngf, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1)

        self.conv4_1 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, dilation=2, padding=2)
        self.conv4_2 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, dilation=2, padding=2)
        self.conv4_3 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, dilation=2, padding=2)

        self.deconv6_1 = nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1)
        self.conv6_2 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, padding=1)
        self.conv6_3 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, padding=1)

        self.deconv7_1 = nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1)
        self.conv7_2 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, padding=1)

        self.deconv8_1 = nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1)
        self.conv8_2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1)

        self.color_pred = nn.Conv2d(ngf, num_outputs, kernel_size=1)
        if Kaiming:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, inputs):
        conv1_1 = self.conv1_1(inputs)
        conv1_2 = self.conv1_2(conv1_1)

        conv2_1 = self.conv2_1(conv1_2)
        conv2_2 = self.conv2_2(conv2_1)

        conv3_1 = self.conv3_1(conv2_2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)

        conv4_1 = self.conv4_1(conv3_3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)

        skip = torch.cat([conv4_3, conv3_3], dim=1)
        deconv6_1 = self.deconv6_1(skip)
        conv6_2 = self.conv6_2(deconv6_1)
        conv6_3 = self.conv6_3(conv6_2)

        skip = torch.cat([conv6_3, conv2_2], dim=1)
        deconv7_1 = self.deconv7_1(skip)
        conv7_2 = self.conv7_2(deconv7_1)

        skip = torch.cat([conv7_2, conv1_2], dim=1)
        deconv8_1 = self.deconv8_1(skip)
        conv8_2 = self.conv8_2(deconv8_1)

        feat = conv8_2
        pred = self.color_pred(feat)

        return pred
    

