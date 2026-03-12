import os
import cv2
import torch
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, root_dir, image_ids, transform=None):
        """
        root_dir : VOC dataset root
        image_ids : list of image IDs
        transform : albumentations transform pipeline
        """

        self.root_dir = root_dir
        self.image_ids = image_ids
        self.transform = transform

        self.images_dir = os.path.join(root_dir, "JPEGImages")
        self.masks_dir = os.path.join(root_dir, "SegmentationClass")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        image_id = self.image_ids[idx]

        img_path = os.path.join(self.images_dir, image_id + ".jpg")
        mask_path = os.path.join(self.masks_dir, image_id + ".png")

        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # ensure mask labels are valid
        mask[mask > 20] = 255

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure mask type is correct
        mask = mask.long()

        return image, mask