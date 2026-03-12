import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, preds, targets):

        preds = F.softmax(preds, dim=1)

        total_dice = 0
        valid_classes = 0

        for cls in range(self.num_classes):

            if cls == self.ignore_index:
                continue

            pred_cls = preds[:, cls]
            target_cls = (targets == cls).float()

            mask = (targets != self.ignore_index).float()

            pred_cls = pred_cls * mask
            target_cls = target_cls * mask

            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()

            dice = (2 * intersection + 1e-6) / (union + 1e-6)

            total_dice += dice
            valid_classes += 1

        return 1 - total_dice / valid_classes


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index)

    def forward(self, preds, targets):

        ce_loss = self.ce(preds, targets)
        dice_loss = self.dice(preds, targets)

        loss = 0.5 * ce_loss + 0.5 * dice_loss

        return loss