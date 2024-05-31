import torch
import numpy as np
import os
from PIL import Image
from torchvision import models, transforms

from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F


class L2pooling(torch.nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        # a = torch.hann_window(5,periodic=False)
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


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

class E_DISTS(torch.nn.Module):
    '''
    Refer to https://github.com/dingkeyan93/DISTS
    '''

    def __init__(self, channels=3):
        assert channels == 3
        super(E_DISTS, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_parameter("alpha", torch.nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_parameter("beta", torch.nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)

        weights = torch.load(os.path.abspath(os.path.join('weights/DISTS.pth')))
        self.alpha.data = weights['alpha']
        self.beta.data = weights['beta']


    def forward_once(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, mask_finder):
        assert x.shape == y.shape

        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y)
        
        masks = []

        for i in range(len(self.chns)):
            masks.append(mask_finder[i](torch.cat([feats0[i], feats1[i]], 1)))
        
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)

        maps_masked = []
        maps = []

        for k in range(len(self.chns)):
            feat0 = feats0[k] * masks[k]
            feat1 = feats1[k] * masks[k]
            
            x_mean = feat0.mean([2, 3], keepdim=True)
            y_mean = feat1.mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feat0 - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feat1 - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feat0 * feat1).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

        score = 1 - (dist1 + dist2).squeeze()
        return score.item()


class E_DISTS_Runner(torch.nn.Module):
    def __init__(self):

        super(E_DISTS_Runner, self).__init__()

        # Init All Components
        self.cuda()

        self.chns = [3, 64, 128, 256, 512, 512]

        self.L = len(self.chns)

        self.mask_finder = []
        self.mask_finder_1 = MaskFinder(self.chns[0] * 2).cuda()
        self.mask_finder_2 = MaskFinder(self.chns[1] * 2).cuda()
        self.mask_finder_3 = MaskFinder(self.chns[2] * 2).cuda()
        self.mask_finder_4 = MaskFinder(self.chns[3] * 2).cuda()
        self.mask_finder_5 = MaskFinder(self.chns[4] * 2).cuda()
        self.mask_finder_6 = MaskFinder(self.chns[4] * 2).cuda()

        self.mask_finder_1.requires_grad = False
        self.mask_finder_2.requires_grad = False
        self.mask_finder_3.requires_grad = False
        self.mask_finder_4.requires_grad = False
        self.mask_finder_5.requires_grad = False
        self.mask_finder_6.requires_grad = False

        self.mask_finder.append(self.mask_finder_1)
        self.mask_finder.append(self.mask_finder_2)
        self.mask_finder.append(self.mask_finder_3)
        self.mask_finder.append(self.mask_finder_4)
        self.mask_finder.append(self.mask_finder_5)
        self.mask_finder.append(self.mask_finder_6)

        self.model_edists = E_DISTS().to(device)

        model_path = os.path.abspath(os.path.join('weights', 'E_DISTS.pth'))
        self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)


    def forward(self, y, x, as_loss=True, resize = True):

        return self.model_edists(x, y, self.mask_finder)


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

    model_edists_runner = E_DISTS_Runner().to(device)
    
    E_DISTS_score = model_edists_runner(dist, ref)
        
    print('Score: ' + str(E_DISTS_score))
