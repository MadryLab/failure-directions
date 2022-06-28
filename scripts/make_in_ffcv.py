import torchvision
import os
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
import torch
import src.pytorch_datasets as pytorch_datasets
from src import ffcv_utils
import yaml
from src.config_parsing import ffcv_read_check_override_config
import pprint
from src.ffcv_utils import get_training_loaders
from src.pytorch_datasets import create_val_split, get_unlabeled_indices, IndexedDataset
import sys
import os
sys.path.append('..')

def get_unlabeled(name, initial_train_targets, num_classes, folds, first_val_split=5):
    # write subsets
    for fold in folds:
        result_indices = get_unlabeled_indices(initial_train_targets=initial_train_targets, 
                                               num_classes=num_classes, fold=fold, first_val_split=first_val_split)
        print("--", fold, "--")
        for k, v in result_indices.items():
            print(k, len(v))
        torch.save(result_indices, f'index_files/{name}_indices_{fold}.pt')
        
BETON_ROOT = "/home/gridsan/sajain/CorrErrs_shared/betons"

def write_betons(ds_name, train_ds, test_ds, val_ds=None, max_resolution=None):
    os.makedirs(os.path.join(BETON_ROOT, ds_name), exist_ok=True)
    ds_pairs = [
        ['train', train_ds],
        ['test', test_ds]
    ]
    if val_ds is not None:
        ds_pairs.append(['val', val_ds])
    
    for split_name, ds in ds_pairs:
        ds = IndexedDataset(ds)
        write_path = os.path.join(BETON_ROOT, ds_name, f"{ds_name}_{split_name}.beton")
        # Pass a type for each data field
        img_field = RGBImageField() if max_resolution is None else RGBImageField(max_resolution=max_resolution)
        writer = DatasetWriter(write_path, {
            # Tune options to optimize dataset size, throughput at train-time
            'image': img_field,
            'label': IntField(),
            'index': IntField(),
        })

        # Write dataset
        writer.from_indexed_dataset(ds)
        
train_ds = torchvision.datasets.ImageFolder("/home/gridsan/groups/datasets/ImageNet/train")
test_ds = torchvision.datasets.ImageFolder("/home/gridsan/groups/datasets/ImageNet/val")
write_betons('imagenet', train_ds, test_ds, val_ds=None, max_resolution=256)

