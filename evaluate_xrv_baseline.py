import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

import torchxrayvision as xrv

import data_utils
from data_utils import NIHChestXrayDataset, get_xrv_preprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained torchxrayvision model on NIH ChestXray14")
    parser.add_argument("--weights", default="densenet121-res224-nih",
                        help="torchxrayvision weights identifier (e.g., densenet121-res224-nih)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional path to a custom state dict to load after initializing weights")
    parser.add_argument("--nih_img_dir", required=True, help="Directory containing NIH images")
    parser.add_argument("--split", default="nih14_val", choices=["nih14_train", "nih14_val"],
                        help="Dataset split")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Optional JSON string or path for per-class thresholds (list or {class:value}).")
    parser.add_argument("--nih_train_fraction", type=float, default=0.9)
    parser.add_argument("--nih_split_seed", type=int, default=0)
    parser.add_argument("--nih_views", type=str, default="PA")
    parser.add_argument("--nih_csv_path", type=str, default=None,
                        help="Path to NIH Data_Entry CSV (defaults to torchxrayvision bundled file)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sweep_thresholds", action="store_true",
                        help="If set, sweep per-class thresholds (maximizing F1) on this split before reporting metrics.")
    parser.add_argument("--sweep_steps", type=int, default=201,
                        help="Number of thresholds between 0 and 1 to evaluate when sweeping.")
    parser.add_argument("--save_thresholds", type=str, default=None,
                        help="Optional path to save the thresholds actually used for evaluation as JSON.")
    parser.add_argument("--pr_output", type=str, default=None,
                        help="Optional JSON path to dump per-class PR curves")
    return parser.parse_args()


def prepare_loader(args):
    views = [v.strip() for v in args.nih_views.split(',') if v.strip()]
    data_utils.configure_nih_dataset(img_dir=args.nih_img_dir,
                                     csv_path=args.nih_csv_path,
                                     train_fraction=args.nih_train_fraction,
                                     split_seed=args.nih_split_seed,
                                     views=views)
    preprocess = get_xrv_preprocess()
    split_name = args.split
    if split_name.startswith("nih14_"):
        split_name = split_name.split('_', 1)[1]
    dataset = NIHChestXrayDataset(split=split_name, preprocess=preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
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
        ordered = []
        for cls in classes:
            if cls not in data:
                raise ValueError(f"Threshold for class '{cls}' missing in provided dict.")
            ordered.append(float(data[cls]))
        return np.asarray(ordered, dtype=np.float32)
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


def sweep_thresholds(probs, targets, steps, metric="f1"):
    if steps < 2:
        raise ValueError("sweep_steps must be >= 2")
    thresholds = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    probs = probs.astype(np.float32)
    targets_bool = targets.astype(bool)
    total = targets.shape[0]

    preds = probs[None, :, :] > thresholds[:, None, None]

    tp = np.logical_and(preds, targets_bool[None, :, :]).sum(axis=1).astype(np.float32)
    fp = np.logical_and(preds, np.logical_not(targets_bool)[None, :, :]).sum(axis=1).astype(np.float32)
    fn = np.logical_and(np.logical_not(preds), targets_bool[None, :, :]).sum(axis=1).astype(np.float32)
    tn = (total - tp - fp - fn).astype(np.float32)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    accuracy = (tp + tn) / max(total, 1)

    metric_scores = f1 if metric == "f1" else precision
    score_label = "f1" if metric == "f1" else "precision"

    valid = (targets_bool.sum(axis=0) > 0) & (targets_bool.sum(axis=0) < total)
    metric_scores[:, ~valid] = -np.inf

    best_idx = np.argmax(metric_scores, axis=0)
    cols = np.arange(probs.shape[1])

    best_thresholds = thresholds[best_idx]
    best_thresholds = np.where(valid, best_thresholds, np.nan)
    best_scores = metric_scores[best_idx, cols]
    best_scores = np.where(valid, best_scores, np.nan)
    best_accs = accuracy[best_idx, cols]
    best_accs = np.where(valid, best_accs, np.nan)
    best_precision = precision[best_idx, cols]
    best_precision = np.where(valid, best_precision, np.nan)
    best_recall = recall[best_idx, cols]
    best_recall = np.where(valid, best_recall, np.nan)

    return best_thresholds, best_scores, best_accs, best_precision, best_recall, score_label


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
    loader, dataset = prepare_loader(args)

    model = xrv.models.get_model(weights=args.weights)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(state, strict=True)
    model = model.to(args.device)
    model.eval()

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

    threshold_values = None
    sweep_scores = None
    sweep_accs = None
    sweep_precisions = None
    sweep_recalls = None
    sweep_label = "f1"
    if args.sweep_thresholds:
        if args.thresholds:
            print("Warning: --thresholds is ignored because --sweep_thresholds is set.")
        display_thresholds, sweep_scores, sweep_accs, sweep_precisions, sweep_recalls, sweep_label = sweep_thresholds(
            probs, targets, args.sweep_steps, metric=args.sweep_metric
        )
        threshold_values = np.where(np.isnan(display_thresholds), args.threshold, display_thresholds)
    elif args.thresholds:
        threshold_values = _load_thresholds(args.thresholds, nih_classes)
    metrics = compute_metrics(probs, targets, args.threshold, nih_classes, threshold_values)

    if threshold_values is None:
        threshold_label = f"{args.threshold}"
    elif args.sweep_thresholds:
        threshold_label = f"per-class(swept-{sweep_label})"
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
        print(f"\nSwept thresholds (best {sweep_label} per class):")
        for idx, cls in enumerate(nih_classes):
            thr = display_thresholds[idx]
            thr_str = f"{thr:.3f}" if thr == thr else "n/a"
            prec = sweep_precisions[idx]
            rec = sweep_recalls[idx]
            acc = sweep_accs[idx]
            sc = sweep_scores[idx]
            f1_val = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) > 0 else float("nan")
            auc = metrics["aurocs"][idx]
            print(
                f"  {cls:20s} thr={thr_str:>5} auc={auc:.4f} "
                f"acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1_val:.4f} "
                f"{sweep_label}={sc:.4f}"
            )

    if args.save_thresholds and threshold_values is not None:
        save_thresholds(args.save_thresholds, nih_classes, threshold_values)
        print(f"\nSaved thresholds to {args.save_thresholds}")
    elif args.save_thresholds:
        print("\nWarning: --save_thresholds specified but no thresholds were computed.")

    if args.pr_output:
        save_pr_curves(args.pr_output, metrics)
        print(f"\nSaved precision-recall curves to {args.pr_output}")


if __name__ == "__main__":
    main()
