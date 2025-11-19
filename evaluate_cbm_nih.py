import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

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
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Optional JSON string or path specifying per-class thresholds "
                             "(list ordered as LABEL_FILES or {class: value}). Overrides --threshold.")
    parser.add_argument("--sweep_thresholds", action="store_true",
                        help="If set, sweep thresholds on this split (per class) to maximize F1 before computing metrics.")
    parser.add_argument("--sweep_steps", type=int, default=201,
                        help="Number of evenly spaced thresholds between 0 and 1 to evaluate when sweeping.")
    parser.add_argument("--save_thresholds", type=str, default=None,
                        help="Optional path to save the thresholds actually used for evaluation as JSON.")
    parser.add_argument("--nih_train_fraction", type=float, default=0.9)
    parser.add_argument("--nih_split_seed", type=int, default=0)
    parser.add_argument("--nih_views", type=str, default="PA", help="Comma separated view list (e.g., 'PA,AP')")
    parser.add_argument("--nih_csv_path", type=str, default=None,
                        help="Path to NIH Data_Entry CSV (defaults to torchxrayvision bundled file)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pr_output", type=str, default=None,
                        help="Optional path to save per-class PR curves as JSON")
    return parser.parse_args()


def load_backbone_name(model_dir):
    args_path = os.path.join(model_dir, "args.txt")
    with open(args_path) as f:
        args = json.load(f)
    backbone = args.get("backbone", "xrv_densenet121-res224-nih")
    return backbone


def prepare_dataset(split, backbone, cfg, num_workers):
    data_utils.configure_nih_dataset(img_dir=cfg["img_dir"],
                                     csv_path=cfg.get("csv_path"),
                                     train_fraction=cfg["train_fraction"],
                                     split_seed=cfg["split_seed"],
                                     views=cfg["views"])
    target_model, preprocess = data_utils.get_target_model(backbone, "cpu")
    del target_model
    dataset = data_utils.get_data(split, preprocess)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, dataset


def _load_thresholds(thresholds_arg, classes):
    if thresholds_arg is None:
        return None
    if os.path.exists(thresholds_arg):
        with open(thresholds_arg, "r") as f:
            data = json.load(f)
    else:
        data = json.loads(thresholds_arg)
    if isinstance(data, dict):
        values = []
        for cls in classes:
            if cls not in data:
                raise ValueError(f"Threshold for class '{cls}' missing in provided dict.")
            values.append(float(data[cls]))
        return np.asarray(values, dtype=np.float32)
    values = np.asarray(data, dtype=np.float32)
    if values.shape[0] != len(classes):
        raise ValueError("Per-class threshold list must match number of classes.")
    return values


def compute_metrics(probs, targets, threshold, classes, per_class_thresholds=None):
    num_classes = probs.shape[1]
    if per_class_thresholds is not None:
        thresholds = per_class_thresholds.reshape(1, num_classes)
    else:
        thresholds = np.full((1, num_classes), threshold, dtype=np.float32)
    preds = (probs > thresholds).astype(np.float32)
    exact_match = (preds == targets).mean()
    micro_precision = ((preds * targets).sum()) / preds.sum() if preds.sum() > 0 else 0.0
    positives = targets.sum()
    micro_recall = ((preds * targets).sum()) / positives if positives > 0 else 0.0

    aurocs = []
    aps = []
    pr_curves = []
    for idx in range(len(classes)):
        y_true = targets[:, idx]
        y_score = probs[:, idx]
        if len(np.unique(y_true)) < 2:
            aurocs.append(float("nan"))
            aps.append(float("nan"))
            pr_curves.append(None)
            continue
        aurocs.append(roc_auc_score(y_true, y_score))
        aps.append(average_precision_score(y_true, y_score))
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_curves.append({"precision": precision.tolist(), "recall": recall.tolist()})

    per_class_accuracy = (preds == targets).mean(axis=0)

    return {
        "exact_match": exact_match,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "mean_auroc": np.nanmean(aurocs),
        "mean_ap": np.nanmean(aps),
        "aurocs": aurocs,
        "aps": aps,
        "per_class_accuracy": per_class_accuracy,
        "pr_curves": pr_curves,
        "classes": classes,
    }


def _f1_score(y_true, preds):
    tp = np.logical_and(preds == 1, y_true == 1).sum()
    fp = np.logical_and(preds == 1, y_true == 0).sum()
    fn = np.logical_and(preds == 0, y_true == 1).sum()
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def _accuracy_score(y_true, preds):
    if y_true.size == 0:
        return float("nan")
    return (preds == y_true).mean()


def sweep_thresholds(probs, targets, classes, steps):
    if steps < 2:
        raise ValueError("sweep_steps must be >= 2")
    grid = np.linspace(0.0, 1.0, steps)
    num_classes = probs.shape[1]
    best_thresholds = np.full(num_classes, 0.5, dtype=np.float32)
    best_scores = np.full(num_classes, -np.inf, dtype=np.float32)
    best_accs = np.full(num_classes, np.nan, dtype=np.float32)
    for idx in range(num_classes):
        y_true = targets[:, idx]
        if len(np.unique(y_true)) < 2:
            continue
        y_score = probs[:, idx]
        for t in grid:
            preds = (y_score > t).astype(np.float32)
            score = _f1_score(y_true, preds)
            if score > best_scores[idx]:
                best_scores[idx] = score
                best_thresholds[idx] = t
                best_accs[idx] = _accuracy_score(y_true, preds)
    best_scores = np.where(best_scores == -np.inf, np.nan, best_scores)
    return best_thresholds, best_scores, best_accs


def save_thresholds(path, classes, values):
    payload = {cls: float(val) for cls, val in zip(classes, values)}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_pr_curves(path, metrics):
    payload = {
        "classes": metrics["classes"],
        "curves": metrics["pr_curves"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    views = [v.strip() for v in args.nih_views.split(',') if v.strip()]
    nih_cfg = {
        "img_dir": args.nih_img_dir,
        "csv_path": args.nih_csv_path,
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

    classes_path = data_utils.LABEL_FILES["nih14"]
    with open(classes_path) as f:
        classes = [c for c in f.read().splitlines() if c]
    threshold_values = None
    sweep_scores = None
    sweep_accs = None
    threshold_label = f"{args.threshold}"
    if args.sweep_thresholds:
        if args.thresholds:
            print("Warning: --thresholds is ignored because --sweep_thresholds is set.")
        threshold_values, sweep_scores, sweep_accs = sweep_thresholds(probs, targets, classes, args.sweep_steps)
        threshold_label = "swept"
    elif args.thresholds:
        threshold_values = _load_thresholds(args.thresholds, classes)
        threshold_label = "loaded"

    metrics = compute_metrics(probs, targets, args.threshold, classes, threshold_values)

    if threshold_values is None:
        threshold_label = f"{args.threshold}"
    elif args.sweep_thresholds:
        threshold_label = "per-class(swept)"
    elif args.thresholds:
        threshold_label = "per-class(custom)"
    else:
        threshold_label = "per-class"

    print("Evaluation summary:")
    print(f"  exact_match: {metrics['exact_match']:.4f}")
    print(f"  micro_precision@{threshold_label}: {metrics['micro_precision']:.4f}")
    print(f"  micro_recall@{threshold_label}: {metrics['micro_recall']:.4f}")
    print(f"  mean_auroc: {metrics['mean_auroc']:.4f}")
    print(f"  mean_ap: {metrics['mean_ap']:.4f}")

    print("\nPer-class metrics:")
    for cls, auc, ap, acc in zip(metrics["classes"], metrics["aurocs"], metrics["aps"], metrics["per_class_accuracy"]):
        print(f"  {cls:20s} auc={auc:.4f} ap={ap:.4f} acc={acc:.4f}")

    if sweep_scores is not None:
        print("\nSwept thresholds (best F1 per class):")
        for cls, thr, score, acc in zip(classes, threshold_values, sweep_scores, sweep_accs):
            score_val = float(score) if score > -np.inf else float("nan")
            acc_val = float(acc) if not np.isnan(acc) else float("nan")
            print(f"  {cls:20s} thr={thr:.3f} f1={score_val:.4f} acc={acc_val:.4f}")

    if args.save_thresholds and threshold_values is not None:
        save_thresholds(args.save_thresholds, classes, threshold_values)
        print(f"\nSaved thresholds to {args.save_thresholds}")
    elif args.save_thresholds:
        print("\nWarning: --save_thresholds specified but no thresholds were computed.")

    if args.pr_output:
        save_pr_curves(args.pr_output, metrics)
        print(f"\nSaved precision-recall curves to {args.pr_output}")


if __name__ == "__main__":
    main()
