import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torchxrayvision as xrv

import data_utils
from data_utils import NIHChestXrayDataset, get_xrv_preprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a torchxrayvision model on NIH ChestXray14")
    parser.add_argument("--weights", default="densenet121-res224-nih",
                        help="torchxrayvision weights identifier, e.g. densenet121-res224-nih")
    parser.add_argument("--nih_img_dir", required=True, help="Path to NIH images (e.g., images-224)")
    parser.add_argument("--nih_train_fraction", type=float, default=0.9)
    parser.add_argument("--nih_split_seed", type=int, default=0)
    parser.add_argument("--nih_views", type=str, default="PA", help="Comma separated view list")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output", required=True,
                        help="Directory to save fine-tuned checkpoint (state_dict.pth and config.json)")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="If set, only fine-tune classifier head")
    return parser.parse_args()


def prepare_loaders(args):
    views = [v.strip() for v in args.nih_views.split(",") if v.strip()]
    data_utils.configure_nih_dataset(img_dir=args.nih_img_dir,
                                     train_fraction=args.nih_train_fraction,
                                     split_seed=args.nih_split_seed,
                                     views=views)
    preprocess = get_xrv_preprocess()
    train_ds = NIHChestXrayDataset(split="train", preprocess=preprocess)
    val_ds = NIHChestXrayDataset(split="val", preprocess=preprocess)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader


def evaluate(model, loader, device, idx_map):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)[:, idx_map]
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())
    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    aurocs = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) < 2:
            aurocs.append(float("nan"))
        else:
            aurocs.append(roc_auc_score(targets[:, i], logits[:, i]))
    return np.nanmean(aurocs)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    train_loader, val_loader = prepare_loaders(args)

    model = xrv.models.get_model(weights=args.weights).to(args.device)
    with open(data_utils.LABEL_FILES["nih14"]) as f:
        nih_classes = [c for c in f.read().splitlines() if c]
    idx_map = []
    for name in nih_classes:
        clean = name.replace("_", " ")
        if name in model.pathologies:
            idx_map.append(model.pathologies.index(name))
        elif clean in model.pathologies:
            idx_map.append(model.pathologies.index(clean))
        else:
            raise ValueError(f"Class '{name}' not found in model pathologies {model.pathologies}")
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -float("inf")
    best_path = os.path.join(args.output, "state_dict.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            logits = model(images)[:, idx_map]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        val_auc = evaluate(model, val_loader, args.device, idx_map)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_mean_auroc={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)
            with open(os.path.join(args.output, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            print(f"Saved best checkpoint with AUROC {best_auc:.4f} to {best_path}")

    print(f"Training complete. Best val AUROC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
