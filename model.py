import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


def get_model(num_classes=21):

    model = deeplabv3_mobilenet_v3_large(
        weights="DEFAULT"
    )

    # Replace classifier for VOC classes
    model.classifier[4] = nn.Conv2d(
        in_channels=256,
        out_channels=num_classes,
        kernel_size=1
    )

    return model