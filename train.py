
import os
import csv
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import VOCDataset
from augmentations import get_train_transforms, get_val_transforms
from model import get_model
from losses import CombinedLoss

torch.backends.cudnn.benchmark = True


def dice_score(preds, targets):

    preds = torch.argmax(preds, dim=1)

    intersection = (preds == targets).float().sum()
    union = preds.numel()

    return (intersection / union).item()


def main():

    # ----------------------------
    # Paths
    # ----------------------------

    DATA_ROOT = "VOC2012_train_val/VOC2012_train_val"
    IDS_FILE = os.path.join(DATA_ROOT, "ImageSets/Segmentation/trainval.txt")

    # ----------------------------
    # Load IDs
    # ----------------------------

    with open(IDS_FILE) as f:
        ids = f.read().splitlines()

    train_ids, val_ids = train_test_split(
        ids,
        test_size=0.2,
        random_state=42
    )

    # ----------------------------
    # Datasets
    # ----------------------------

    train_dataset = VOCDataset(
        DATA_ROOT,
        train_ids,
        transform=get_train_transforms()
    )

    val_dataset = VOCDataset(
        DATA_ROOT,
        val_ids,
        transform=get_val_transforms()
    )

    # ----------------------------
    # DataLoaders
    # ----------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ----------------------------
    # Model
    # ----------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=21).to(device)

    # ----------------------------
    # Loss
    # ----------------------------

    criterion = CombinedLoss(num_classes=21)

    # ----------------------------
    # Optimizer
    # ----------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    # ----------------------------
    # AMP
    # ----------------------------

    scaler = torch.amp.GradScaler("cuda")

    # ----------------------------
    # Logging dictionary
    # ----------------------------

    log = {
        "epoch": [],
        "train_loss": [],
        "val_dice": []
    }

    # ----------------------------
    # Training Loop
    # ----------------------------

    best_dice = 0

    for epoch in range(30):

        model.train()

        train_loss = 0

        for images, masks in tqdm(train_loader):

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):

                outputs = model(images)["out"]

                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch} Train Loss:", train_loss)

        # ----------------------------
        # Validation
        # ----------------------------

        model.eval()

        val_dice = 0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)["out"]

                val_dice += dice_score(outputs, masks)

        val_dice /= len(val_loader)

        print("Validation Dice:", val_dice)

        # ----------------------------
        # Save logs
        # ----------------------------

        log["epoch"].append(epoch)
        log["train_loss"].append(train_loss)
        log["val_dice"].append(val_dice)

        # ----------------------------
        # Save best model
        # ----------------------------

        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(model.state_dict(), "best_model.pth")

            print("Best model saved")

    # ----------------------------
    # Save CSV log
    # ----------------------------

    with open("training_log.csv", "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_dice"])

        for i in range(len(log["epoch"])):

            writer.writerow([
                log["epoch"][i],
                log["train_loss"][i],
                log["val_dice"][i]
            ])


if __name__ == "__main__":
    main()