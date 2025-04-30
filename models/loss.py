import torch
import torch.nn as nn


class DetailRestorationLoss(nn.Module):
    """Loss function for detail restoration branch"""

    def __init__(self, epsilon=1e-3):
        super(DetailRestorationLoss, self).__init__()
        self.epsilon = epsilon

    def charbonnier_loss(self, pred, target):
        """Charbonnier Loss
        L_r = sqrt(||I' - Î||_2 + ε^2)
        where ε = 10^-3
        """
        return torch.sqrt(torch.mean((pred - target) ** 2 + self.epsilon ** 2))

    def forward(self, pred, target):
        """Calculate total loss
        Args:
            pred: Predicted image
            target: Target image (ground truth)
        Returns:
            loss: Charbonnier loss value
        """
        return self.charbonnier_loss(pred, target)


class ColorRestorationLoss(nn.Module):
    """Loss function for color restoration branch"""

    def __init__(self):
        super(ColorRestorationLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        """Calculate color restoration loss
        Args:
            pred: Predicted image
            target: Target image
        Returns:
            loss: L1 loss value
        """
        return self.l1_loss(pred, target)


class TotalLoss(nn.Module):
    """Total loss function"""

    def __init__(self, branch_weight=[0.5, 0.5]):
        super(TotalLoss, self).__init__()
        self.detail_loss = DetailRestorationLoss()
        self.color_loss = ColorRestorationLoss()
        self.detail_weight = branch_weight[0]
        self.color_weight = branch_weight[1]

    def forward(self, detail_pred, color_pred, target):
        """Calculate total loss
        Args:
            detail_pred: Prediction from detail restoration branch
            color_pred: Prediction from color restoration branch
            target: Target image
        Returns:
            total_loss: Total loss value
            losses: Dictionary containing individual losses
        """
        # Calculate branch losses
        detail_loss = self.detail_loss(detail_pred, target)
        color_loss = self.color_loss(color_pred, target)

        # Calculate total loss
        total_loss = self.detail_weight * detail_loss + self.color_weight * color_loss

        # Return total loss and individual losses
        losses = {
            'total': total_loss,
            'detail': detail_loss,
            'color': color_loss
        }

        return total_loss, losses
