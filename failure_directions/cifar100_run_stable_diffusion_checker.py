import sys
import os
sys.path.append('..')
import torch
import torchvision
import failure_directions
import numpy as np
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import failure_directions.src.svm_utils as svm_utils
import failure_directions.src.visualization_utils as viz_utils
import failure_directions.src.ds_utils as ds_utils
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from failure_directions.src.label_maps import CLASS_DICT
import pickle as pkl
from torchvision.datasets.folder import pil_loader
from failure_directions.src.config_parsing import ffcv_read_check_override_config
import yaml
from collections import defaultdict
import src.stable_diffusion_utils as sd_utils
from src.stable_diffusion_utils import DiffDataset
import failure_directions.src.pytorch_datasets as pytorch_datasets

import torchvision.transforms as transforms
import torchvision
from failure_directions.src.decoders_and_transforms import PyTranslate, PyCutOut
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-trials",
    type=int,
    default=15,
)

parser.add_argument(
    "--one-class",
    type=int,
    default=-1,
)
parser.add_argument(
    "--out-file",
    type=str,
    default="dump.pt"
)
INVALID_CLASSES = [1, 12, 13]

opt = parser.parse_args()
num_trials = opt.num_trials
one_class = opt.one_class
out_file = opt.out_file

#Load SVM Model
beton_root = "/mnt/cfs/projects/correlated_errors/betons"
experiment_root = "/mnt/cfs/projects/correlated_errors/experiments/spurious_cifar100/unlabeled_1_4_new_spurious_norm"

svm_name = "svm_spurious_unlabeled_normalized"
name = os.path.join(experiment_root, f"svm_checkpoints/{svm_name}.pt") # SVM output file
svm_model_name = os.path.join(experiment_root, f"svm_checkpoints/{svm_name}_model.pkl") # SVM output file
model_root = os.path.join(experiment_root, "models")
model_ckpt = os.path.join(model_root, "spurious_supercifar100_unlabeled/version_0/checkpoints/checkpoint_last.pt")
loss_upweight_root = os.path.join(experiment_root, "loss_vec_files")
subset_root = os.path.join(experiment_root, "subset_index_files")

processor = viz_utils.SVMProcessor(name, root=beton_root, checkpoint_path=model_ckpt, get_unlabeled=True)
classes_to_drop = torch.load(processor.metrics['args']['indices_file'])['classes_to_drop']

svm_model = processor._build_model(model_ckpt)

split = 'test'
test_dv = processor.metrics[f'{split}_metrics']['decision_values']
test_confs = processor.run_dict[split]['confs']
test_superclass = processor.metrics[f'{split}_metrics']['classes'] # 0 if female, 1 if male
test_subclass = processor.metrics[f'{split}_metrics']['spuriouses'] #1 if blond, 2 if black hair, 0 if neither
test_problematic = np.in1d(test_subclass, classes_to_drop)
test_pred_correct = processor.metrics[f'{split}_metrics']['ypred']
test_correct = processor.metrics[f'{split}_metrics']['ytrue']


with open(processor.metrics['args']['config'], 'r') as file:
    fresh_hparams = yaml.safe_load(file)
fresh_hparams = ffcv_read_check_override_config(fresh_hparams)

images_path = "/mnt/cfs/home/saachij/src/stable-diffusion/cifar100_images"
path_dict = sd_utils.get_path_dict(images_path, num_classes=20)


hparams = processor.hparams

# 
fill_color = tuple(map(int, hparams['mean']))

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=np.array(hparams['mean'])/255, std=np.array(hparams['std'])/255)])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    PyTranslate(2),
    PyCutOut(4, fill_color),
    base_transform
])

resize_base_transform = transforms.Compose([base_transform, transforms.Resize((32, 32))])
resize_train_transform = transforms.Compose([train_transform, transforms.Resize((32, 32))])

# 
def evaluate_model(model, loader):
    with torch.no_grad():
        with autocast():
            gts, preds, confs = [], [], []
            for x, y in tqdm(loader):
                x = x.cuda()
                logits = model(x)
                gts.append(y.cpu())
                preds.append(logits.argmax(-1).cpu())
                softmax_logits = nn.Softmax(dim=-1)(logits)
                confs.append(softmax_logits[torch.arange(logits.shape[0]), y].cpu())
    gts = torch.cat(gts)
    preds = torch.cat(preds)
    confs = torch.cat(confs)
    return gts, preds, confs

# 
class SuperCIFAR100Wrapper:
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        x, y, _ = self.ds[idx]
        return x, y

bsz = fresh_hparams['batch_size']
ds_root = "/mnt/cfs/datasets/cifar100"

orig_train_ds = SuperCIFAR100Wrapper(pytorch_datasets.SuperCIFAR100(root=ds_root, train=True, transform=base_transform))
aug_train_ds = SuperCIFAR100Wrapper(pytorch_datasets.SuperCIFAR100(root=ds_root, train=True, transform=train_transform))
test_ds = SuperCIFAR100Wrapper(pytorch_datasets.SuperCIFAR100(root=ds_root, train=False, transform=base_transform))

val_indices = processor.indices_dict['val_indices']
train_indices = processor.indices_dict['train_indices']

train_ds = torch.utils.data.Subset(aug_train_ds, train_indices)
val_ds = torch.utils.data.Subset(orig_train_ds, val_indices)
no_aug_train_ds = torch.utils.data.Subset(orig_train_ds, train_indices)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bsz, shuffle=True, drop_last=True)
no_shuffle_train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bsz, shuffle=False, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=bsz, shuffle=False, drop_last=False)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bsz, shuffle=False, drop_last=False)
no_aug_train_loader = torch.utils.data.DataLoader(no_aug_train_ds, batch_size=bsz, shuffle=False, drop_last=False)


# 
def run_model(train_loader_, val_loader_, test_loader_, set_device=False):
    build_fn = failure_directions.model_utils.BUILD_FUNCTIONS[hparams['arch_type']]
    model = build_fn(hparams['arch'], hparams['num_classes'])
    model = model.cuda()

    training_args=hparams['training']
    training_args['iters_per_epoch'] = len(train_loader_)
    trainer = failure_directions.LightWeightTrainer(training_args=hparams['training'],
                                                    exp_name='temp', enable_logging=False,
                                                    bce=False, set_device=set_device)
    trainer.fit(model, train_loader_, val_loader_)
    return evaluate_model(model, test_loader_)

# See relative accuracies
flip_interventions = {}
for intensity in [0, 0.1, 0.2, 0.3]:
    if opt.one_class != -1:
        print("only one class")
        include_classes = [opt.one_class]
    else:
        print("all classes")
        include_classes = [c for c in np.arange(hparams['num_classes']) if c not in INVALID_CLASSES]
    base_synth_dataset = DiffDataset(path_dict, flip_name='flip', intensity=intensity, num_imgs_per_class=100,
                                     transform=resize_train_transform, include_classes=include_classes,
                                     num_classes=hparams['num_classes'])
    synth_train_set = torch.utils.data.ConcatDataset([train_ds, base_synth_dataset])
    synth_train_loader = torch.utils.data.DataLoader(synth_train_set, batch_size=bsz, shuffle=True, drop_last=True)
    flip_interventions[intensity] = [
        run_model(synth_train_loader, val_loader, test_loader, set_device=True)
        for _ in range (num_trials)
    ]
    
torch.save(flip_interventions, out_file)
