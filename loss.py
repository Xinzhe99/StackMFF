# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()
        # Define Laplacian operator convolution kernel
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]], dtype=torch.float32)
        # Expand dimensions to adapt to image channels
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)

    def forward(self, pred, target):
        # Ensure kernel is on the correct device
        if pred.is_cuda:
            self.kernel = self.kernel.cuda()

        # If input is multi-channel image, need to process each channel
        if len(pred.shape) == 4:  # B x C x H x W
            b, c, h, w = pred.shape
            # Expand kernel to corresponding channel count
            kernel = self.kernel.expand(c, 1, 3, 3)
            
            # Ensure kernel is on the correct device
            if pred.is_cuda:
                kernel = kernel.cuda()

            # Calculate Laplacian gradient of predicted image
            pred_lap = F.conv2d(pred, kernel, padding=1, groups=c)
            # Calculate Laplacian gradient of target image
            target_lap = F.conv2d(target, kernel, padding=1, groups=c)

        else:  # Single channel image B x H x W
            # Ensure kernel is on the correct device
            if pred.is_cuda:
                self.kernel = self.kernel.cuda()
                
            pred_lap = F.conv2d(pred.unsqueeze(1), self.kernel, padding=1)
            target_lap = F.conv2d(target.unsqueeze(1), self.kernel, padding=1)

        # Calculate L1 loss
        loss = torch.mean(torch.abs(pred_lap - target_lap))
        return loss


class SpatialFrequencyLoss(nn.Module):
    def __init__(self, kernel_radius=5):
        super(SpatialFrequencyLoss, self).__init__()
        self.kernel_radius = kernel_radius

    def cal_sf(self, x):
        device = x.device
        b, c, h, w = x.shape

        # Define shift convolution kernels for horizontal and vertical directions
        r_shift_kernel = torch.FloatTensor([[0, 0, 0],
                                            [1, 0, 0],
                                            [0, 0, 0]]).to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)

        b_shift_kernel = torch.FloatTensor([[0, 1, 0],
                                            [0, 0, 0],
                                            [0, 0, 0]]).to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)

        # Calculate shifts in horizontal and vertical directions
        x_r_shift = F.conv2d(x, r_shift_kernel, padding=1, groups=c)
        x_b_shift = F.conv2d(x, b_shift_kernel, padding=1, groups=c)

        # Calculate gradients
        x_grad = torch.pow((x_r_shift - x), 2) + torch.pow((x_b_shift - x), 2)

        # Calculate spatial frequency
        kernel_size = self.kernel_radius * 2 + 1
        add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().to(device)
        kernel_padding = kernel_size // 2
        x_sf = torch.sum(F.conv2d(x_grad, add_kernel, padding=kernel_padding, groups=c), dim=1, keepdim=True)

        return x_sf

    def forward(self, pred, target):
        # Calculate spatial frequency of predicted and target images
        pred_sf = self.cal_sf(pred)
        target_sf = self.cal_sf(target)

        # Calculate spatial frequency loss (using L1 loss)
        sf_loss = torch.mean(torch.abs(pred_sf - target_sf))

        return sf_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
    
    
class LpLssimLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        """
            Constructor
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)  # 使用已定义的函数

    def gaussian(self, window_size, sigma):
        """
            Get the gaussian kernel which will be used in SSIM computation
        """
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        """
            Create the gaussian window
        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)   # [window_size, 1]
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0) # [1,1,window_size, window_size]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
            Compute the SSIM for the given two image
            The original source is here: https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
        """
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, image_in, image_out):

        # Check if need to create the gaussian window
        (_, channel, _, _) = image_in.size()
        if channel == self.channel and self.window.data.type() == image_in.data.type():
            pass
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(image_out.get_device()) if image_out.is_cuda else window
            window = window.type_as(image_in)
            self.window = window
            self.channel = channel
        Lssim = 1 - self._ssim(image_in, image_out, self.window, self.window_size, self.channel, self.size_average)
        return Lssim
