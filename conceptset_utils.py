import requests
import random
import numpy as np
import torch
import math
from tqdm import tqdm
import clip
from sentence_transformers import SentenceTransformer    
    
def get_init_conceptnet(classes, limit=200, relations=["HasA", "IsA", "PartOf", "HasProperty", "MadeOf", "AtLocation"]):
    concepts = set()

    for cls in tqdm(classes):
        words = cls.replace(',', '').split(' ')
        for word in words:
            obj = requests.get('http://api.conceptnet.io/c/en/{}?limit={}'.format(word, limit)).json()
            obj.keys()
            for dicti in obj['edges']:
                rel = dicti['rel']['label']
                try:
                    if dicti['start']['language'] != 'en' or dicti['end']['language'] != 'en':
                        continue
                except(KeyError):
                    continue

                if rel in relations:
                    if rel in ["IsA"]: 
                        concepts.add(dicti['end']['label'])
                    else:
                        concepts.add(dicti['start']['label'])
                        concepts.add(dicti['end']['label'])
    return concepts


def remove_too_long(concepts, max_len, print_prob=0):
    """
    deletes all concepts longer than max_len
    """
    new_concepts = []
    for concept in concepts:
        if len(concept) <= max_len:
            new_concepts.append(concept)
        else:
            if random.random()<print_prob:
                print(len(concept), concept)
    print(len(concepts), len(new_concepts))
    return new_concepts


def filter_too_similar_to_cls(concepts, classes, sim_cutoff, device="cuda", print_prob=0):
    #first check simple text matches
    print(len(concepts))
    concepts = list(concepts)
    concepts = sorted(concepts)
    
    for cls in classes:
        for prefix in ["", "a ", "A ", "an ", "An ", "the ", "The "]:
            try:
                concepts.remove(prefix+cls)
                if random.random()<print_prob:
                    print("Class:{} - Deleting {}".format(cls, prefix+cls))
            except(ValueError):
                pass
        try:
            concepts.remove(cls.upper())
        except(ValueError):
            pass
        try:
            concepts.remove(cls[0].upper()+cls[1:])
        except(ValueError):
            pass
    print(len(concepts))
        
    mpnet_model = SentenceTransformer('all-mpnet-base-v2')
    class_features_m = mpnet_model.encode(classes)
    concept_features_m = mpnet_model.encode(concepts)
    dot_prods_m = class_features_m @ concept_features_m.T
    dot_prods_c = _clip_dot_prods(classes, concepts)
    #weighted since mpnet has highger variance
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    
    to_delete = []
    for i in range(len(classes)):
        for j in range(len(concepts)):
            prod = dot_prods[i,j]
            if prod >= sim_cutoff and i!=j:
                if j not in to_delete:
                    to_delete.append(j)
                    if random.random()<print_prob:
                        print("Class:{} - Concept:{}, sim:{:.3f} - Deleting {}".format(classes[i], concepts[j], dot_prods[i,j], concepts[j]))
                        print("".format(concepts[j]))
                        
    to_delete = sorted(to_delete)[::-1]

    for item in to_delete:
        concepts.pop(item)
    print(len(concepts))
    return concepts

def filter_too_similar(concepts, sim_cutoff, device="cuda", print_prob=0):
    
    mpnet_model = SentenceTransformer('all-mpnet-base-v2')
    concept_features = mpnet_model.encode(concepts)
        
    dot_prods_m = concept_features @ concept_features.T
    dot_prods_c = _clip_dot_prods(concepts, concepts)
    
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    
    to_delete = []
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            prod = dot_prods[i,j]
            if prod >= sim_cutoff and i!=j:
                if i not in to_delete and j not in to_delete:
                    to_print = random.random() < print_prob
                    #Deletes the concept with lower average similarity to other concepts - idea is to keep more general concepts
                    if np.sum(dot_prods[i]) < np.sum(dot_prods[j]):
                        to_delete.append(i)
                        if to_print:
                            print("{} - {} , sim:{:.4f} - Deleting {}".format(concepts[i], concepts[j], dot_prods[i,j], concepts[i]))
                    else:
                        to_delete.append(j)
                        if to_print:
                            print("{} - {} , sim:{:.4f} - Deleting {}".format(concepts[i], concepts[j], dot_prods[i,j], concepts[j]))
                            
    to_delete = sorted(to_delete)[::-1]
    for item in to_delete:
        concepts.pop(item)
    print(len(concepts))
    return concepts


def _clip_dot_prods(list1, list2, device="cuda", clip_name="ViT-B/16", batch_size=500):
    "Returns: numpy array with dot products"
    clip_model, _ = clip.load(clip_name, device=device)
    text1 = clip.tokenize(list1).to(device)
    text2 = clip.tokenize(list2).to(device)
    
    features1 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text1)/batch_size)):
            features1.append(clip_model.encode_text(text1[batch_size*i:batch_size*(i+1)]))
        features1 = torch.cat(features1, dim=0)
        features1 /= features1.norm(dim=1, keepdim=True)

    features2 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text2)/batch_size)):
            features2.append(clip_model.encode_text(text2[batch_size*i:batch_size*(i+1)]))
        features2 = torch.cat(features2, dim=0)
        features2 /= features2.norm(dim=1, keepdim=True)
        
    dot_prods = features1 @ features2.T
    return dot_prods.cpu().numpy()

def most_similar_concepts(word, concepts, device="cuda"):
    """
    returns most similar words to a given concepts
    """
    mpnet_model = SentenceTransformer('all-mpnet-base-v2')
    word_features = mpnet_model.encode([word])
    concept_features = mpnet_model.encode(concepts)
        
    dot_prods_m = word_features @ concept_features.T
    dot_prods_c = _clip_dot_prods([word], concepts, device)
    
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    min_distance, indices = torch.topk(torch.FloatTensor(dot_prods[0]), k=5)
    return [(concepts[indices[i]], min_distance[i]) for i in range(len(min_distance))]