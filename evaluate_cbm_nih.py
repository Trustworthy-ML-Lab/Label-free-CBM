import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import cbm
import data_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CBM on the NIH ChestXray14 dataset")
    parser.add_argument("--model_dir", required=True, help="Directory containing the trained CBM (args.txt, W_c.pt, etc.)")
    parser.add_argument("--nih_img_dir", required=True, help="Path to NIH images (e.g., images-224)")
    parser.add_argument("--split", default="nih14_val", choices=["nih14_train", "nih14_val"], help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for multilabel metrics")
    parser.add_argument("--nih_train_fraction", type=float, default=0.9)
    parser.add_argument("--nih_split_seed", type=int, default=0)
    parser.add_argument("--nih_views", type=str, default="PA", help="Comma separated view list (e.g., 'PA,AP')")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def load_backbone_name(model_dir):
    args_path = os.path.join(model_dir, "args.txt")
    with open(args_path) as f:
        args = json.load(f)
    backbone = args.get("backbone", "xrv_densenet121-res224-nih")
    return backbone


def prepare_dataset(split, backbone, cfg, num_workers):
    data_utils.configure_nih_dataset(img_dir=cfg["img_dir"],
                                     train_fraction=cfg["train_fraction"],
                                     split_seed=cfg["split_seed"],
                                     views=cfg["views"])
    # get preprocess for backbone (instantiate on cpu to save memory)
    target_model, preprocess = data_utils.get_target_model(backbone, "cpu")
    del target_model
    dataset = data_utils.get_data(split, preprocess)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, dataset


def main():
    args = parse_args()
    views = [v.strip() for v in args.nih_views.split(',') if v.strip()]
    nih_cfg = {
        "img_dir": args.nih_img_dir,
        "train_fraction": args.nih_train_fraction,
        "split_seed": args.nih_split_seed,
        "views": views,
        "batch_size": args.batch_size,
    }

    backbone = load_backbone_name(args.model_dir)

    loader, dataset = prepare_dataset(args.split, backbone, nih_cfg, args.num_workers)

    model = cbm.load_cbm(args.model_dir, args.device)
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(args.device)
            logits, _ = model(images)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_targets.append(labels.float())

    probs = torch.cat(all_probs, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    preds = (probs > args.threshold).astype(np.float32)
    exact_match = (preds == targets).mean()
    micro_f1 = f1_score(targets.reshape(-1), preds.reshape(-1))

    classes_path = data_utils.LABEL_FILES["nih14"]
    with open(classes_path) as f:
        classes = [c for c in f.read().splitlines() if c]

    aurocs = []
    ap_scores = []
    for c in range(len(classes)):
        y_true = targets[:, c]
        y_score = probs[:, c]
        if len(np.unique(y_true)) < 2:
            aurocs.append(float("nan"))
            ap_scores.append(float("nan"))
            continue
        aurocs.append(roc_auc_score(y_true, y_score))
        ap_scores.append(average_precision_score(y_true, y_score))

    summary = {
        "exact_match": exact_match,
        "micro_f1": micro_f1,
        "mean_auroc": np.nanmean(aurocs),
        "mean_ap": np.nanmean(ap_scores),
    }

    print("Evaluation summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    print("\nPer-class AUROC / AP:")
    for cls, auc, ap in zip(classes, aurocs, ap_scores):
        print(f"  {cls:20s} auc={auc:.4f} ap={ap:.4f}")


if __name__ == "__main__":
    main()
