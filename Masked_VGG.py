import torch
import numpy as np
import os
from PIL import Image
from torchvision import models, transforms

from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

class MaskFinder(torch.nn.Module):
    def __init__(self, input_channels, num_features=64):
        super(MaskFinder, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channels, out_channels=num_features, stride=1, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True))

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=num_features, out_channels=num_features, stride=1, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True))

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=num_features, out_channels=1, stride=1, kernel_size=3, padding=1))

        self.sigmoid = torch.nn.Sigmoid()
        self.softplus = torch.nn.Softplus()

    def forward(self, inputChannels):
        out_conv1 = self.conv1(inputChannels)
        out_conv2 = self.conv2(out_conv1)
        out = self.conv3(out_conv2)
        out = self.sigmoid(out)

        return out

class EVGG(torch.nn.Module):
    def __init__(self):
        super(EVGG, self).__init__()

        # Init All Components

        self.chns = [64, 128, 256, 512, 512]

        self.L = len(self.chns)

        self.mask_finder = []
        self.mask_finder_1 = MaskFinder(self.chns[0] * 2).cuda()
        self.mask_finder_2 = MaskFinder(self.chns[1] * 2).cuda()
        self.mask_finder_3 = MaskFinder(self.chns[2] * 2).cuda()
        self.mask_finder_4 = MaskFinder(self.chns[3] * 2).cuda()
        self.mask_finder_5 = MaskFinder(self.chns[4] * 2).cuda()

        self.mask_finder_1.requires_grad = False
        self.mask_finder_2.requires_grad = False
        self.mask_finder_3.requires_grad = False
        self.mask_finder_4.requires_grad = False
        self.mask_finder_5.requires_grad = False

        self.mask_finder.append(self.mask_finder_1)
        self.mask_finder.append(self.mask_finder_2)
        self.mask_finder.append(self.mask_finder_3)
        self.mask_finder.append(self.mask_finder_4)
        self.mask_finder.append(self.mask_finder_5)

        vgg_pretrained_features = models.vgg16(pretrained=True).features
        # self.stage1 = torch.nn.Sequential()
        # self.stage2 = torch.nn.Sequential()
        # self.stage3 = torch.nn.Sequential()
        # self.stage4 = torch.nn.Sequential()
        # self.stage5 = torch.nn.Sequential()
        # for x in range(0, 4):
        #     self.stage1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(4, 9):
        #     self.stage2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(9, 16):
        #     self.stage3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(16, 23):
        #     self.stage4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(23, 30):
        #     self.stage5.add_module(str(x), vgg_pretrained_features[x])

        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.chns = [64, 128, 256, 512, 512]

        model_path = os.path.abspath(os.path.join('weights', 'E_VGG.pth'))
        self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

        self.cuda()


    def forward_once(self, x):
        h = self.stage1(x)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        outs_normalized = []
        for k in range(len(outs)):
            outs_normalized.append(F.normalize(outs[k]))
        return outs, outs_normalized

    def forward(self, x, y, as_loss=False):
        assert x.shape == y.shape
        with torch.no_grad():
            feats0_non, feats0 = self.forward_once(x)
            feats1_non, feats1 = self.forward_once(y)
        masks = []
        for i in range(len(self.chns)):
            masks.append(self.mask_finder[i](torch.cat([feats0_non[i], feats1_non[i]], 1)))

        ssims = torch.zeros(1, np.sum(self.chns)).cuda()
        counter = 0

        for k in range(len(self.chns)):
            ssim = (masks[k] * torch.abs(feats0_non[k] - feats1_non[k])).mean([2, 3], keepdim=True)
            ssims[:, counter:counter + self.chns[k]] = ssims[:, counter:counter + self.chns[k]] + ssim.reshape(1, self.chns[k])
            counter = counter + self.chns[k]
        val = (torch.sum(ssims, dim=1) / np.sum(self.chns)).reshape(1, 1, 1, 1)
        return val.mean()


def prepare_image(image, resize = False, repeatNum = 1):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0).repeat(repeatNum,1,1,1)
    
        
if __name__ == '__main__':

    import argparse
    from data import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='images/ref.BMP')
    parser.add_argument('--dist', type=str, default='images/dist.BMP')
    parser.add_argument('--save_dir', type=str, default='save_map')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image(Image.open(args.ref).convert("RGB")).to(device)
    dist = prepare_image(Image.open(args.dist).convert("RGB")).to(device)

    model_evgg = EVGG().to(device)
    
    E_VGG_score = model_evgg(dist, ref)
        
    print('Score: ' + str(E_VGG_score))
