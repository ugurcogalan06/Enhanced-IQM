import torch
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms
# import matplotlib.pyplot as plt

class ScalerNetwork(torch.nn.Module):
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(ScalerNetwork, self).__init__()

        layers = [torch.nn.Conv2d(1, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [torch.nn.LeakyReLU(0.2,True),]
        layers += [torch.nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [torch.nn.LeakyReLU(0.2,True),]
        layers += [torch.nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [torch.nn.Sigmoid(),]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, val):
        return self.model.forward(val)


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
        self.relu = torch.relu

    def forward(self, inputChannels):
        out_conv1 = self.conv1(inputChannels)
        out_conv2 = self.conv2(out_conv1)
        out = self.conv3(out_conv2)
        out = self.sigmoid(out)

        return out


class E_PSNR(torch.nn.Module):
    def __init__(self):

        super(E_PSNR, self).__init__()

        # Init All Components
        self.cuda()

        self.chns = [3]

        self.L = len(self.chns)

        self.mask_finder = []
        self.mask_finder_1 = MaskFinder(self.chns[0] * 2).cuda()

        self.mask_finder_1.requires_grad = False

        self.scaler_network = ScalerNetwork()

        model_path = os.path.abspath(os.path.join('weights', 'E_PSNR.pth'))
        self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)


    def forward(self, y, x, as_loss=True, resize = True):
        mask = self.mask_finder_1(torch.cat([x, y], 1))
        score, mse = self.psnr(mask * x * 255, mask * y * 255)

        return score


    def psnr(self, img1, img2):
        squared_error = (img1 - img2) ** 2
        mse = squared_error.mean([1, 2, 3], keepdim=True)
        return 20 * torch.log10(255.0 / torch.sqrt(mse)), squared_error


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

    model_epsnr = E_PSNR().to(device)

    E_PSNR_score = model_epsnr(dist, ref)
    
    print('Score: ' + str(E_PSNR_score))


   