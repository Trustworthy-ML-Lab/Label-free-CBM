import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import torchxrayvision as xrv

import data_utils
from data_utils import NIHChestXrayDataset, get_xrv_preprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained torchxrayvision model on NIH ChestXray14")
    parser.add_argument("--weights", default="densenet121-res224-nih",
                        help="torchxrayvision weights identifier (e.g., densenet121-res224-nih)")
    parser.add_argument("--nih_img_dir", required=True, help="Directory containing NIH images")
    parser.add_argument("--split", default="nih14_val", choices=["nih14_train", "nih14_val"],
                        help="Dataset split")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nih_train_fraction", type=float, default=0.9)
    parser.add_argument("--nih_split_seed", type=int, default=0)
    parser.add_argument("--nih_views", type=str, default="PA")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def prepare_loader(args):
    views = [v.strip() for v in args.nih_views.split(',') if v.strip()]
    data_utils.configure_nih_dataset(img_dir=args.nih_img_dir,
                                     train_fraction=args.nih_train_fraction,
                                     split_seed=args.nih_split_seed,
                                     views=views)
    preprocess = get_xrv_preprocess()
    split_name = args.split
    if split_name.startswith("nih14_"):
        split_name = split_name.split("_", 1)[1]
    dataset = NIHChestXrayDataset(split=split_name, preprocess=preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    return loader, dataset


def compute_metrics(probs, targets, threshold):
    preds = (probs > threshold).astype(np.float32)
    exact_match = (preds == targets).mean()
    micro_f1 = f1_score(targets.reshape(-1), preds.reshape(-1))

    classes_path = data_utils.LABEL_FILES["nih14"]
    with open(classes_path) as f:
        classes = [c for c in f.read().splitlines() if c]

    aurocs = []
    aps = []
    for idx in range(len(classes)):
        y_true = targets[:, idx]
        y_score = probs[:, idx]
        if len(np.unique(y_true)) < 2:
            aurocs.append(float("nan"))
            aps.append(float("nan"))
            continue
        aurocs.append(roc_auc_score(y_true, y_score))
        aps.append(average_precision_score(y_true, y_score))

    return {
        "exact_match": exact_match,
        "micro_f1": micro_f1,
        "mean_auroc": np.nanmean(aurocs),
        "mean_ap": np.nanmean(aps),
        "aurocs": aurocs,
        "aps": aps,
        "classes": classes,
    }


def main():
    args = parse_args()
    loader, dataset = prepare_loader(args)

    model = xrv.models.get_model(weights=args.weights)
    model = model.to(args.device)
    model.eval()

    # Map torchxrayvision pathologies to NIH class order
    with open(data_utils.LABEL_FILES["nih14"]) as f:
        nih_classes = [c for c in f.read().splitlines() if c]
    idx_map = []
    for name in nih_classes:
        clean = name.replace('_', ' ')
        if name in model.pathologies:
            idx_map.append(model.pathologies.index(name))
        elif clean in model.pathologies:
            idx_map.append(model.pathologies.index(clean))
        else:
            raise ValueError(f"Class '{name}' not found in model pathologies {model.pathologies}")

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating XRV model"):
            images = images.to(args.device)
            logits = model(images)
            probs = torch.sigmoid(logits)[:, idx_map]
            all_probs.append(probs.cpu())
            all_targets.append(labels.float())

    probs = torch.cat(all_probs, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    metrics = compute_metrics(probs, targets, args.threshold)

    print("Evaluation summary:")
    for k in ("exact_match", "micro_f1", "mean_auroc", "mean_ap"):
        print(f"  {k}: {metrics[k]:.4f}")

    print("\nPer-class AUROC / AP:")
    for cls, auc, ap in zip(metrics["classes"], metrics["aurocs"], metrics["aps"]):
        print(f"  {cls:20s} auc={auc:.4f} ap={ap:.4f}")


if __name__ == "__main__":
    main()
