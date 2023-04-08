import os
import argparse
import datetime
import json
import torch

import clip
import utils
import data_utils

from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser(description='Settings for creating model')

parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--feature_layer", type=str, default='layer4', 
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
parser.add_argument("--lam", type=float, default=0.0125, help="Sparsity regularization parameter, higher->more sparse")
parser.add_argument("--n_iters", type=int, default=1000, help="How many iterations to run the final layer solver for")

def train_and_save(args):
    #load data and models
    
    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"
    
    target_model, target_preprocess = data_utils.get_target_model(args.backbone, args.device)

    data_t = data_utils.get_data(d_train, preprocess=target_preprocess)
    val_data_t = data_utils.get_data(d_val, preprocess=target_preprocess)
    
    with open(data_utils.LABEL_FILES[args.dataset], "r") as f:
        classes = f.read().split("\n")
    
    target_save_name, _, _ = utils.get_save_names("", args.backbone, args.feature_layer, d_train, "", "avg",
                                                  args.activation_dir)
    val_target_save_name, _, _ =  utils.get_save_names("", args.backbone, args.feature_layer, d_val, "", "avg",
                                                       args.activation_dir)
    #save activations and get save_paths
    if args.backbone.startswith("clip_"):
        model, _ = clip.load(args.backbone[5:], device=args.device)
        utils.save_clip_image_features(model, data_t, target_save_name, args.batch_size, args.device)
        utils.save_clip_image_features(model, val_data_t, val_target_save_name, args.batch_size, args.device)
    else:
        utils.save_target_activations(target_model, data_t, target_save_name, target_layers = [args.feature_layer],
                                  batch_size = args.batch_size, device = args.device, pool_mode='avg')
        utils.save_target_activations(target_model, val_data_t, val_target_save_name, target_layers = [args.feature_layer],
                                  batch_size = args.batch_size, device = args.device, pool_mode='avg')
    

    #load features
    target_features = torch.load(target_save_name, map_location="cpu").float()
    val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
    
    with torch.no_grad():
        train_c = target_features.detach()
        val_c = val_target_features.detach()
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)

        train_z = (train_c-train_mean)/train_std
        labels = data_t.targets
        train_y = torch.LongTensor(labels)

        indexed_train_ds = IndexedTensorDataset(train_z,train_y)

        val_z = (val_c-train_mean)/train_std
        val_labels = val_data_t.targets
        val_y = torch.LongTensor(val_labels)

        val_ds = TensorDataset(val_z,val_y)


    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_z.shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
                           val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), 
                           n_classes = len(classes))
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    W_g = W_g.to(args.device)
    
    save_name = "{}/{}_finetuned_{}".format(args.save_dir, args.dataset, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    os.mkdir(save_name)
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))
    
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {}
        for key in ('lam', 'lr', 'alpha', 'time'):
            out_dict[key] = float(output_proj['path'][0][key])
        out_dict['metrics'] = output_proj['path'][0]['metrics']
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        json.dump(out_dict, f, indent=2)
    
if __name__=='__main__':
    args = parser.parse_args()
    train_and_save(args)