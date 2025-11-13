import os
import math
import torch
import data_utils
from clip_loader import load_clip_encoder

from tqdm import tqdm
from torch.utils.data import DataLoader


def save_target_features(target_model, dataset, save_name, batch_size=512, device="cuda", num_workers=8):
    _make_save_dir(save_name)
    if os.path.exists(save_name):
        return
    feats = []
    target_model.eval()
    with torch.no_grad():
        for images, _ in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)):
            features = target_model(images.to(device))
            feats.append(features.cpu())
    torch.save(torch.cat(feats), save_name)
    del feats
    torch.cuda.empty_cache()
    return

def save_clip_image_features(clip_wrapper, dataset, save_name, batch_size=1000 , device = "cuda", num_workers=8):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)):
            features = clip_wrapper.encode_image(images.to(device))
            all_features.append(features.cpu())
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(clip_wrapper, texts, save_name, batch_size=1000, device="cuda"):
    
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(texts)/batch_size))):
            batch = texts[batch_size*i:batch_size*(i+1)]
            tokens = clip_wrapper.tokenize(batch).to(device)
            feats = clip_wrapper.encode_text(tokens)
            text_features.append(feats.cpu())
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def save_activations(clip_name, target_name, d_probe, concept_set, batch_size, device, save_dir, num_workers=8):
    target_save_name, clip_save_name, text_save_name = get_save_names(clip_name, target_name,
                                                                     d_probe, concept_set, save_dir)
    save_names = {"clip": clip_save_name, "text": text_save_name, "target": target_save_name}
    if _all_saved(save_names):
        return
    
    clip_wrapper = load_clip_encoder(clip_name, device)
    clip_preprocess = clip_wrapper.preprocess
    
    target_model, target_preprocess = data_utils.get_target_model(target_name, device)
    #setup data
    data_c = data_utils.get_data(d_probe, clip_preprocess)
    data_t = data_utils.get_data(d_probe, target_preprocess)

    with open(concept_set, 'r') as f: 
        words = [line for line in f.read().split('\n') if len(line)]
    text_prompts = ["a chest radiograph showing {}".format(word) for word in words]
    
    save_clip_text_features(clip_wrapper, text_prompts, text_save_name, batch_size, device)
    
    save_clip_image_features(clip_wrapper, data_c, clip_save_name, batch_size, device, num_workers=num_workers)
    save_target_features(target_model, data_t, target_save_name, batch_size, device, num_workers=num_workers)
    
    return
    
    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True):
    image_features = torch.load(clip_save_name)
    text_features = torch.load(text_save_name)
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True).float()
        text_features /= text_features.norm(dim=-1, keepdim=True).float()
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()
    
    target_feats = torch.load(target_save_name)
    similarity = similarity_fn(clip_feats, target_feats)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity
    
def get_save_names(clip_name, target_name, d_probe, concept_set, save_dir):
    target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    clip_save_name = "{}/{}_clip_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split('/')[-1]).split('.')[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    return target_save_name, clip_save_name, text_save_name

    
def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

def get_accuracy_cbm(model, dataset, device, batch_size=250, num_workers=2, multilabel=False, threshold=0.5):
    correct = 0
    total = 0
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                           pin_memory=True)):
        with torch.no_grad():
            #outs = target_model(images.to(device))
            outs, _ = model(images.to(device))
            if multilabel:
                probs = torch.sigmoid(outs)
                pred = (probs > threshold).float().cpu()
                labels = labels.float()
                correct += torch.sum((pred == labels).float())
                total += labels.numel()
            else:
                pred = torch.argmax(outs, dim=1)
                correct += torch.sum(pred.cpu()==labels)
                total += len(labels)
    return correct/total

def get_preds_cbm(model, dataset, device, batch_size=250, num_workers=2, multilabel=False, threshold=0.5):
    preds = []
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                           pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            if multilabel:
                pred = (torch.sigmoid(outs) > threshold).float()
            else:
                pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    return preds

def get_concept_act_by_pred(model, dataset, device):
    preds = []
    concept_acts = []
    for images, labels in tqdm(DataLoader(dataset, 500, num_workers=8, pin_memory=True)):
        with torch.no_grad():
            outs, concept_act = model(images.to(device))
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)
    concept_acts_by_pred=[]
    for i in range(torch.max(pred)+1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds==i], dim=0))
    concept_acts_by_pred = torch.stack(concept_acts_by_pred, dim=0)
    return concept_acts_by_pred
