import argparse
import importlib
import importlib.util
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
    parser = argparse.ArgumentParser(description="Fine-tune a torchxrayvision model on chest X-ray datasets")
    parser.add_argument("--dataset", type=str, default="nih14", choices=["nih14", "chex"],
                        help="Dataset to fine-tune on")
    parser.add_argument("--weights", default="densenet121-res224-nih",
                        help="torchxrayvision weights identifier, e.g. densenet121-res224-nih")
    parser.add_argument("--custom_model", default=None,
                        help="Python spec (<module>:<callable> or /path/to/file.py:callable) returning an "
                             "nn.Module or (nn.Module, preprocess). Overrides --weights when set.")
    parser.add_argument("--custom_model_args", default=None,
                        help="JSON string or path describing keyword args passed to the custom callable.")
    parser.add_argument("--custom_checkpoint", default=None,
                        help="Optional checkpoint to load into the custom model.")
    parser.add_argument("--custom_preprocess", default=None,
                        help="Python callable spec that returns a torchvision transform to preprocess the data. "
                             "Ignored if the custom model callable returns the transform directly.")
    parser.add_argument("--custom_class_list", default=None,
                        help="Path to newline separated class names describing the custom model output order.")
    parser.add_argument("--nih_img_dir", type=str, default=None, help="Path to NIH images (e.g., images-224)")
    parser.add_argument("--nih_train_fraction", type=float, default=0.9)
    parser.add_argument("--nih_split_seed", type=int, default=0)
    parser.add_argument("--nih_views", type=str, default="PA", help="Comma separated view list")
    parser.add_argument("--nih_csv_path", type=str, default=None,
                        help="Path to NIH Data_Entry CSV (defaults to torchxrayvision bundled file)")
    parser.add_argument("--chex_img_dir", type=str, default=None,
                        help="Path to CheXpert images (required for chex dataset)")
    parser.add_argument("--chex_train_fraction", type=float, default=0.9)
    parser.add_argument("--chex_split_seed", type=int, default=0)
    parser.add_argument("--chex_views", type=str, default="PA", help="Comma separated view list for CheXpert")
    parser.add_argument("--chex_csv_path", type=str, default=None,
                        help="Path to CheXpert CSV (defaults to torchxrayvision bundled file)")
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
    parser.add_argument("--pos_weight_mode", type=str, default="none", choices=["none", "balanced", "manual"],
                        help="Class-imbalance handling for BCE loss. 'balanced' uses neg/pos ratios from the train split.")
    parser.add_argument("--pos_weight_file", type=str, default=None,
                        help="Optional JSON path containing a list of manual pos_weight values (length must equal #classes).")
    return parser.parse_args()


def _load_callable(spec):
    if ":" not in spec:
        raise ValueError(f"Callable spec '{spec}' must be in <module or file>:<attr> format.")
    module_path, attr = spec.split(":", 1)
    if module_path.endswith(".py") and os.path.exists(module_path):
        module_name = f"_custom_module_{abs(hash(module_path))}"
        spec_obj = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec_obj)
        assert spec_obj.loader is not None
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)
    if not hasattr(module, attr):
        raise AttributeError(f"{spec}: attribute '{attr}' not found")
    return getattr(module, attr)


def _load_json_arg(value):
    if value is None:
        return {}
    if os.path.exists(value):
        with open(value, "r") as fh:
            return json.load(fh)
    return json.loads(value)


def prepare_loaders(args, preprocess):
    dataset_key = args.dataset
    if dataset_key == "nih14":
        if args.nih_img_dir is None:
            raise ValueError("--nih_img_dir is required for NIH14 dataset")
        views = [v.strip() for v in args.nih_views.split(",") if v.strip()]
        data_utils.configure_nih_dataset(img_dir=args.nih_img_dir,
                                         csv_path=args.nih_csv_path,
                                         train_fraction=args.nih_train_fraction,
                                         split_seed=args.nih_split_seed,
                                         views=views)
    elif dataset_key == "chex":
        if args.chex_img_dir is None:
            raise ValueError("--chex_img_dir is required for CheX dataset")
        views = [v.strip() for v in args.chex_views.split(",") if v.strip()]
        data_utils.configure_chex_dataset(img_dir=args.chex_img_dir,
                                          csv_path=args.chex_csv_path,
                                          train_fraction=args.chex_train_fraction,
                                          split_seed=args.chex_split_seed,
                                          views=views)
    else:
        raise ValueError(f"Unsupported dataset {dataset_key}")

    train_ds = data_utils.get_data(f"{dataset_key}_train", preprocess=preprocess)
    val_ds = data_utils.get_data(f"{dataset_key}_val", preprocess=preprocess)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds


def _select_logits(logits, idx_map):
    if idx_map is None:
        return logits
    return logits[:, idx_map]


def evaluate(model, loader, device, idx_map):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = _select_logits(model(images), idx_map)
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


def _read_class_list(path):
    with open(path, "r") as fh:
        return [line.strip() for line in fh if line.strip()]


def _normalize_name(name):
    return name.replace("_", " ").lower()


def _build_idx_map(target_classes, model_classes):
    if model_classes is None:
        return None
    mapping = []
    normalized = {cls: idx for idx, cls in enumerate(model_classes)}
    normalized_lower = {_normalize_name(cls): idx for idx, cls in enumerate(model_classes)}
    for name in target_classes:
        candidates = [
            name,
            name.replace("_", " "),
            _normalize_name(name),
            _normalize_name(name.replace("_", " "))
        ]
        found_idx = None
        for cand in candidates:
            if cand in normalized:
                found_idx = normalized[cand]
                break
            if cand in normalized_lower:
                found_idx = normalized_lower[cand]
                break
        if found_idx is None:
            raise ValueError(f"Class '{name}' not found in provided model output labels {model_classes}")
        mapping.append(found_idx)
    return mapping


def _instantiate_model(args):
    preprocess = None
    model_classes = None
    if args.custom_model:
        builder = _load_callable(args.custom_model)
        builder_kwargs = _load_json_arg(args.custom_model_args)
        result = builder(**builder_kwargs)
        if isinstance(result, tuple):
            model, preprocess = result
        else:
            model = result
        if not isinstance(model, nn.Module):
            raise TypeError("Custom model callable must return an nn.Module or (nn.Module, preprocess)")
        if args.custom_checkpoint:
            state = torch.load(args.custom_checkpoint, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"Loaded custom checkpoint with missing keys {missing} and unexpected keys {unexpected}")
        if preprocess is None:
            if args.custom_preprocess:
                preprocess_fn = _load_callable(args.custom_preprocess)
                preprocess = preprocess_fn()
            else:
                preprocess = get_xrv_preprocess()
        if args.custom_class_list:
            model_classes = _read_class_list(args.custom_class_list)
        elif hasattr(model, "pathologies"):
            model_classes = list(model.pathologies)
    else:
        model = xrv.models.get_model(weights=args.weights)
        preprocess = get_xrv_preprocess()
        if hasattr(model, "pathologies"):
            model_classes = list(model.pathologies)

    model = model.to(args.device)
    return model, preprocess, model_classes


def _load_manual_pos_weight(path, num_classes):
    with open(path, "r") as f:
        data = json.load(f)
    values = np.asarray(data, dtype=np.float32)
    if values.shape[0] != num_classes:
        raise ValueError("Manual pos_weight list must match number of classes.")
    return torch.from_numpy(values)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    label_file = data_utils.LABEL_FILES[args.dataset]
    with open(label_file) as f:
        dataset_classes = [c for c in f.read().splitlines() if c]

    model, preprocess, model_classes = _instantiate_model(args)
    idx_map = _build_idx_map(dataset_classes, model_classes)

    train_loader, val_loader, train_dataset = prepare_loaders(args, preprocess)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    pos_weight_tensor = None
    if args.pos_weight_mode != "none":
        if args.pos_weight_mode == "manual":
            if args.pos_weight_file is None:
                raise ValueError("--pos_weight_file is required when pos_weight_mode='manual'")
            pos_weight_tensor = _load_manual_pos_weight(args.pos_weight_file, len(dataset_classes))
        else:
            labels = train_dataset.targets
            pos_counts = labels.sum(dim=0)
            neg_counts = labels.shape[0] - pos_counts
            pos_counts = torch.clamp(pos_counts, min=1.0)
            pos_weight_tensor = neg_counts / pos_counts
        print("Using pos_weight:", pos_weight_tensor.tolist())
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(args.device))
    else:
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
            logits = _select_logits(model(images), idx_map)
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
