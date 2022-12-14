import os
import sys
from collections import defaultdict
from torchvision.datasets.folder import pil_loader
import torch
import torchvision
import numpy as np
import tqdm

def abs_walk(directory):
    return [(p, os.path.join(directory, p)) for p in os.listdir(directory)]

def get_path_dict(images_path, num_classes):
    path_dict = {}
    for c in range(num_classes):
        path_dict[c] = {
            "flip": defaultdict(list),
            "no_flip": defaultdict(list),
        }

    for hash_name, hash_path in abs_walk(images_path):
        for cls, cls_path in abs_walk(hash_path):
            for flip_name, flip_path in abs_walk(cls_path):
                sample_path = os.path.join(flip_path, "samples")
                for image_name, image_path in abs_walk(sample_path):
                    intensity = float(image_name.split("_")[1].split(".png")[0])
                    path_dict[int(cls)][flip_name][intensity].append(image_path)
    return path_dict

class DiffDataset:
    def __init__(self, path_dict, intensity, flip_name="flip", num_classes=10, transform=None, 
                 num_imgs_per_class=None, cache_images=True, include_classes=None):
        self.classes = []
        self.img_paths = []
        for c in range(num_classes):
            if include_classes is not None and c not in include_classes:
                continue
            arr = path_dict[c][flip_name][intensity]
            if num_imgs_per_class is not None:
                arr = arr[:num_imgs_per_class]
                assert len(arr) == num_imgs_per_class
            self.classes += [c]*len(arr)
            self.img_paths += arr
        if cache_images:
            self.imgs = [pil_loader(img_path) for img_path in self.img_paths]
        else:
            self.imgs = None
        self.transform = transform
        
    def __len__(self):
        return len(self.classes)
    
    def __getitem__(self, idx):
        c = self.classes[idx]
        img_path = self.img_paths[idx]
        if self.imgs is not None:
            img = self.imgs[idx]
        else:
            img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, c
    
def get_sd_dict(images_path, num_classes):
    path_dict = {}
    for c in range(num_classes):
        path_dict[c] = []
    
    for hash_name, hash_path in abs_walk(images_path):
        for img_name, img_path in abs_walk(hash_path):
            c = int(img_name.split('.')[0]) % 10
            path_dict[c].append(img_path)
    return path_dict

class SDDataset:
    def __init__(self, path_dict, name, num_classes=10, transform=None, 
                 num_imgs_per_class=None, cache_images=True, include_classes=None):
        self.classes = []
        self.img_paths = []
        for c in range(num_classes):
            if include_classes is not None and c not in include_classes:
                continue
            arr = path_dict[name][c]
            if num_imgs_per_class is not None:
                arr = arr[:num_imgs_per_class]
                assert len(arr) == num_imgs_per_class
            self.classes += [c]*len(arr)
            self.img_paths += arr
        if cache_images:
            self.imgs = [pil_loader(img_path) for img_path in tqdm.tqdm(self.img_paths)]
        else:
            self.imgs = None
        self.transform = transform
        
    def __len__(self):
        return len(self.classes)
    
    def __getitem__(self, idx):
        c = self.classes[idx]
        img_path = self.img_paths[idx]
        if self.imgs is not None:
            img = self.imgs[idx]
        else:
            img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, c
        
