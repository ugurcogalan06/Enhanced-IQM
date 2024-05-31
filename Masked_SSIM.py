import torch
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
# import matplotlib.pyplot as plt

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    c = 0.00001

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    sigma1 = torch.sqrt(sigma1_sq + c)
    sigma2 = torch.sqrt(sigma2_sq + c)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2

    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    ssim_map = luminance * contrast * structure
    ssim_map = 1 - ssim_map

    return ssim_map


def ssim(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel)


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


class SSIM(torch.nn.Module):
    def __init__(self):

        super(SSIM, self).__init__()

        # Init All Components
        self.cuda()

        self.scaler_network = ScalerNetwork()

        model_path = os.path.abspath(os.path.join('weights', 'SSIM.pth'))
        self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)


    def forward(self, y, x, as_loss=True, resize = True):

        score = ssim(x, y) 
        
        return score.mean(dim=(1,2,3)).reshape(B, 1, 1, 1)
    
    
    def SSIM_map(self, y, x):
        
        return self.scaler_network(ssim(x, y).mean(dim=(1), keepdim=True)) - self.scaler_network(torch.tensor(0.0).cuda().reshape(1,1,1,1))


class E_SSIM(torch.nn.Module):
    def __init__(self):

        super(E_SSIM, self).__init__()

        # Init All Components
        self.cuda()

        self.chns = [3]

        self.L = len(self.chns)

        self.mask_finder = []
        self.mask_finder_1 = MaskFinder(self.chns[0] * 2).cuda()

        self.mask_finder_1.requires_grad = False

        self.scaler_network = ScalerNetwork()

        model_path = os.path.abspath(os.path.join('weights', 'E_SSIM.pth'))
        self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)


    def forward(self, y, x, as_loss=True, resize = True):

        mask = self.mask_finder_1(torch.cat([x, y], 1))

        score = ssim(x * masks[0], y * masks[0])

        return score.mean(dim=(1,2,3)).reshape(B, 1, 1, 1)
        

    def E_SSIM_map(self, y, x):

        C, H, W = x.shape[0:3]
        
        masks = self.mask_finder_1(torch.cat([x, y], 1))

        return self.scaler_network(ssim(x * masks[0], y * masks[0]).mean(dim=(1), keepdim=True)) - self.scaler_network(torch.tensor(0.0).cuda().reshape(1,1,1,1)), masks[0]


def prepare_image(image, resize = False, repeatNum = 1):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0).repeat(repeatNum,1,1,1)

def sigmoid_scaling(input):
	return torch.abs(1 - (2 / (1 + torch.exp(25 * input))))

def map_visualization(input):
    
    input= input.detach().squeeze().cpu().numpy()
    input = CHWtoHWC(index2color(np.round(input * 255.0), get_magma_map()))

    return input

    
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

    model_emae = E_SSIM().to(device)
    model_mae = SSIM().to(device)

    E_SSIM_err, E_SSIM_mask = model_emae.E_SSIM_map(dist, ref)
    SSIM_err = model_mae.SSIM_map(dist, ref)
    
    E_SSIM_err = map_visualization(E_SSIM_err)
    SSIM_err = map_visualization(SSIM_err)

    # plt.imshow(dist.squeeze().permute([1,2,0]).cpu().numpy())
    # plt.show()
    # plt.imshow(ref.squeeze().permute([1,2,0]).cpu().numpy())
    # plt.show()
    # plt.imshow(MAE_err[:,:,3::-1])
    # plt.show()
    # plt.imshow(E_MAE_err[:,:,3::-1])
    # plt.show()
    E_SSIM_mask = E_SSIM_mask.detach().squeeze().cpu().numpy()
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_image(os.path.join(args.save_dir, 'E_SSIM_map.png'), E_SSIM_err)
    save_image(os.path.join(args.save_dir, 'SSIM_map.png'), SSIM_err)
    save_image(os.path.join(args.save_dir, 'Mask.png'), E_SSIM_mask)


   