import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (a differentiable variant of L1 loss)"""
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon ** 2)
        return loss.mean()

class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features"""
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # 取 relu3_3 层输出
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # 归一化到VGG输入范围
        mean = torch.tensor([0.485, 0.456, 0.406]).to(pred.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(pred.device).view(1,3,1,1)
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        pred_features = self.vgg_layers(pred_norm)
        target_features = self.vgg_layers(target_norm)
        loss = F.l1_loss(pred_features, target_features)
        return loss

class DetailLoss(nn.Module):
    def __init__(self, lambda_vgg=1.0, epsilon=1e-3):
        super(DetailLoss, self).__init__()
        self.charbonnier = CharbonnierLoss(epsilon)
        self.vgg_loss = VGGPerceptualLoss()
        self.lambda_vgg = lambda_vgg

    def forward(self, detail_pred, ground_truth):
        l_r = self.charbonnier(detail_pred, ground_truth)
        l_vgg = self.vgg_loss(detail_pred, ground_truth)
        return l_r + self.lambda_vgg * l_vgg



