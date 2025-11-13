import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv

LABEL_FILES = {"nih14": "data/nih14_classes.txt"}
MULTILABEL_DATASETS = {"nih14"}

NIH_CFG = {
    "img_dir": os.environ.get("NIH_CXR_IMG_DIR"),
    "train_fraction": 0.9,
    "split_seed": 0,
    "views": ("PA",)
}


def get_data(dataset_name, preprocess=None):
    if dataset_name.startswith("nih14_"):
        split = dataset_name.split("_", 1)[1]
        if split not in ("train", "val"):
            raise ValueError("NIH dataset only supports 'train' and 'val' splits")
        return NIHChestXrayDataset(split=split, preprocess=preprocess)
    raise ValueError(f"Unknown dataset {dataset_name}")


def get_targets_only(dataset_name):
    dataset = get_data(dataset_name)
    return dataset.targets


def get_target_model(target_name, device):
    if not target_name.startswith("xrv_"):
        raise ValueError("Only torchxrayvision backbones are supported. Use names like 'xrv_densenet121-res224-nih'")
    weight_key = target_name[4:]
    backbone = xrv.models.get_model(weights=weight_key)
    encoder = TorchXRayEncoder(backbone).to(device)
    encoder.eval()
    preprocess = get_xrv_preprocess()
    return encoder, preprocess


def configure_nih_dataset(img_dir=None, train_fraction=None, split_seed=None, views=None):
    if img_dir is not None:
        NIH_CFG["img_dir"] = img_dir
    if train_fraction is not None:
        if not (0 < train_fraction < 1):
            raise ValueError("train_fraction must be between 0 and 1")
        NIH_CFG["train_fraction"] = float(train_fraction)
    if split_seed is not None:
        NIH_CFG["split_seed"] = int(split_seed)
    if views is not None and len(views) > 0:
        NIH_CFG["views"] = tuple(views)


def get_xrv_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])


class NIHChestXrayDataset(Dataset):
    def __init__(self, split, preprocess=None):
        if NIH_CFG["img_dir"] is None:
            raise ValueError("Set NIH_CXR_IMG_DIR or pass --nih_img_dir to point to the NIH images.")
        self.split = split
        self.preprocess = preprocess
        self.base = xrv.datasets.NIH_Dataset(imgpath=NIH_CFG["img_dir"],
                                             views=list(NIH_CFG["views"]),
                                             unique_patients=True,
                                             transform=None,
                                             data_aug=None)
        self.indices = self._select_indices(len(self.base), split,
                                            NIH_CFG["train_fraction"],
                                            NIH_CFG["split_seed"])
        labels = self.base.labels[self.indices]
        self.targets = torch.from_numpy(labels.copy()).float()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = int(self.indices[idx])
        sample = self.base[base_idx]
        img = self._to_pil(sample["img"])
        if self.preprocess is not None:
            img_tensor = self.preprocess(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        label = torch.from_numpy(sample["lab"]).float()
        return img_tensor, label

    @staticmethod
    def _select_indices(total, split, train_fraction, seed):
        if not (0 < train_fraction < 1):
            raise ValueError("train_fraction must be between 0 and 1")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(total)
        cutoff = int(total * train_fraction)
        if cutoff <= 0 or cutoff >= total:
            raise ValueError("train_fraction results in empty train/val split")
        if split == "train":
            return perm[:cutoff]
        if split == "val":
            return perm[cutoff:]
        raise ValueError(f"Unsupported NIH split '{split}'")

    @staticmethod
    def _to_pil(img):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        img = img.astype(np.float32)
        img -= img.min()
        denom = img.max()
        if denom > 0:
            img /= denom
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img).convert("RGB")


class TorchXRayEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected a 4D tensor (N,C,H,W)")
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        feats = self.backbone.features(x)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        return feats.view(feats.size(0), -1)
