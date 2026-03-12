from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

model = deeplabv3_mobilenet_v3_large(weights="DEFAULT")
print(model.classifier)