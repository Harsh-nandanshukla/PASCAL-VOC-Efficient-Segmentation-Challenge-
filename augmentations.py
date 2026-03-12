# import albumentations as A
# from albumentations.pytorch import ToTensorV2


# def get_train_transforms():
#     return A.Compose([
        
#         # Resize to required input size
#         A.Resize(300, 300),

#         # Geometric augmentations
#         A.HorizontalFlip(p=0.5),
#         A.RandomScale(scale_limit=0.5, p=0.5),
#         A.RandomResizedCrop(300, 300, scale=(0.5, 1.0), p=0.5),
        

#         # Robustness augmentations
#         A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
#         A.GaussianBlur(blur_limit=(3,7), p=0.2),

#         # Lighting changes
#         A.ColorJitter(
#             brightness=0.3,
#             contrast=0.3,
#             saturation=0.3,
#             hue=0.1,
#             p=0.3
#         ),

#         # Compression artifacts
#         A.ImageCompression(
#             quality_range=(40,100),
#             p=0.3
#         ),

#         # Normalize for pretrained MobileNet
#         A.Normalize(
#             mean=(0.485, 0.456, 0.406),
#             std=(0.229, 0.224, 0.225)
#         ),

#         # Convert to tensor
#         ToTensorV2()

#     ])
    

# def get_val_transforms():
#     return A.Compose([
        
#         A.Resize(300, 300),

#         A.Normalize(
#             mean=(0.485, 0.456, 0.406),
#             std=(0.229, 0.224, 0.225)
#         ),

#         ToTensorV2()

#     ])
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    return A.Compose([

        A.RandomResizedCrop(
            size=(300,300),
            scale=(0.5,1.0),
            p=1.0
        ),

        A.HorizontalFlip(p=0.5),

        A.GaussNoise(
            std_range=(0.02,0.1),
            p=0.3
        ),

        A.GaussianBlur(
            blur_limit=(3,7),
            p=0.2
        ),

        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
            p=0.3
        ),

        A.ImageCompression(
            quality_range=(40,100),
            p=0.3
        ),

        A.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225)
        ),

        ToTensorV2()

    ])


def get_val_transforms():
    return A.Compose([

        A.Resize(300,300),

        A.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225)
        ),

        ToTensorV2()

    ])