import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19, resnet18, ResNet18_Weights
from utils.metrics import calculate_ssim


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return loss.mean()


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # 取 relu3_3 层输出
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 归一化到VGG输入范围
        mean = torch.tensor([0.485, 0.456, 0.406]).to(pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(pred.device).view(1, 3, 1, 1)
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        pred_features = self.vgg_layers(pred_norm)
        target_features = self.vgg_layers(target_norm)
        loss = F.l1_loss(pred_features, target_features)
        return loss


class ResNetPerceptualLoss(nn.Module):
    def __init__(self):
        super(ResNetPerceptualLoss, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 取到第二个卷积块结束
        self.resnet_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )
        for param in self.resnet_layers.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 归一化到ResNet输入范围
        mean = torch.tensor([0.485, 0.456, 0.406]).to(pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(pred.device).view(1, 3, 1, 1)
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        pred_features = self.resnet_layers(pred_norm)
        target_features = self.resnet_layers(target_norm)
        loss = F.l1_loss(pred_features, target_features)
        return loss


class DetailVGGLoss(nn.Module):
    def __init__(self, lambda_vgg=1.0, epsilon=1e-3):
        super(DetailVGGLoss, self).__init__()
        self.charbonnier = CharbonnierLoss(epsilon)
        self.vgg_loss = VGGPerceptualLoss()
        self.lambda_vgg = lambda_vgg

    def forward(self, detail_pred: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        l_r = self.charbonnier(detail_pred, ground_truth)
        l_vgg = self.vgg_loss(detail_pred, ground_truth)
        return l_r + self.lambda_vgg * l_vgg


class DetailResNetLoss(nn.Module):
    def __init__(self, lambda_resnet=1.0, epsilon=1e-3):
        super(DetailResNetLoss, self).__init__()
        self.charbonnier = CharbonnierLoss(epsilon)
        self.resnet_loss = ResNetPerceptualLoss()
        self.lambda_resnet = lambda_resnet

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_r = self.charbonnier(pred, target)
        l_resnet = self.resnet_loss(pred, target)
        return l_r + self.lambda_resnet * l_resnet


class DetailSimpleLoss(nn.Module):
    def __init__(self):
        super(DetailSimpleLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        return l1 + 0.5 * mse


class DiffusionLoss(nn.Module):
    def __init__(self):
        super(DiffusionLoss, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.TV_loss = TVLoss()

    def forward(self, output: dict, gt: torch.Tensor) -> torch.Tensor:
        input_high0, input_high1, gt_high0, gt_high1 = output["input_high0"], output["input_high1"], \
            output["gt_high0"], output["gt_high1"]

        pred_LL, gt_LL, pred_x, noise_output, e = output["pred_LL"], output["gt_LL"], output["pred_x"], \
            output["noise_output"], output["e"]

        # =============noise loss==================
        noise_loss = self.l2_loss(noise_output, e)

        # =============frequency loss==================
        frequency_loss = 0.1 * (self.l2_loss(input_high0, gt_high0) +
                                self.l2_loss(input_high1, gt_high1) +
                                self.l2_loss(pred_LL, gt_LL)) +\
            0.01 * (self.TV_loss(input_high0) +
                    self.TV_loss(input_high1) +
                    self.TV_loss(pred_LL))

        # =============photo loss==================
        content_loss = self.l1_loss(pred_x, gt)
        ssim_loss = 1 - calculate_ssim(pred_x, gt, gt.device)

        photo_loss = content_loss + ssim_loss

        return noise_loss, photo_loss, frequency_loss

class LowLightLoss(nn.Module):
    def __init__(self, weight=(0.5, 0.5)):
        super().__init__()
        self.detail_loss = DetailSimpleLoss()
        self.diffusion_loss = DiffusionLoss()
        self.weight = weight

    def forward(self, output, gt: torch.Tensor) -> torch.Tensor:
        # Extract the detail and diffusion outputs from the model output
        detail_output, diffusion_output = output

        # Calculate the detail loss
        detail_loss = self.detail_loss(detail_output, gt)

        # Calculate the diffusion loss
        noise_loss, photo_loss, frequency_loss = self.diffusion_loss(diffusion_output, gt)

        # Combine the losses
        total_loss = self.weight[0] * detail_loss + \
                     self.weight[1] * (noise_loss + photo_loss + frequency_loss)

        return total_loss, detail_loss, noise_loss, photo_loss, frequency_loss