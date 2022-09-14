import torch
import numpy as np
import failure_directions.src.ffcv_utils as ffcv_utils
import failure_directions.src.ffcv_utils as ffcv_utils
from failure_directions.src.config_parsing import ffcv_read_check_override_config
import yaml
import pprint
import tqdm

def get_all_beton_labels(config, split, root, include_spurious=False):
    # get labels from the entire beton
    with open(config, 'r') as file:
        hparams = yaml.safe_load(file)
    hparams = ffcv_read_check_override_config(hparams)

    pipeline = ['label']
    if include_spurious:
        pipeline.append('spurious')
        
    loader = ffcv_utils.get_records(hparams, split, None, pipeline_subset=pipeline, relative_index=True,
                   root=root, indices_dict=None)
    outs = [[] for _ in pipeline]
    for batch in tqdm.tqdm(loader):
        for b in range(len(batch)):
            outs[b].append(batch[b].cpu())
    for b in range(len(outs)):
        outs[b] = torch.cat(outs[b]).numpy()
    if include_spurious:
        return hparams, outs[0], outs[1]
    else:
        return hparams, outs[0]

def _split_dataset(targets, num_classes, split_amt=5):
    # split the dataset into two parts. in Part I: take every split_amt data-point per class.
    # in part II: put the rest of the points
    #
    # Params: 
    #        targets: numpy array of target labels
    #        num_classes: number of classes
    #        split_amt: how much to split into part 1 from the dataset (will take 1/split_amt)
    # Return:
    #        p1_indices: indices belonging to part 1
    #        p2_indices: indices belonging to part 2
    #
    if torch.is_tensor(targets):
        targets = targets.numpy()
    N = len(targets)
    p1_indices = []
    for c in range(num_classes):
        p1_indices.append(np.arange(N)[targets == c][::split_amt])
    p1_indices = np.concatenate(p1_indices)
    p2_indices = np.arange(N)[~np.in1d(np.arange(N), p1_indices)]
    assert len(p1_indices) + len(p2_indices) == N
    return torch.tensor(p1_indices), torch.tensor(p2_indices)
    
def create_dataset_split(train_targets, num_classes, val_split_amt=None, unlabeled_split_amt=None):
    # split the training dataset into train, val, unlabeled
    # NOTE: val is split first, and then unlabeled. 
    # So if val_split_amt is 5, and unlabeled_split_amt is 2 and there are 100 data points
    # val set will have 20 data points, unlabeled will have 40, train will have 40
    #
    # Params:
    #        train_targets: torch tensor of the targets for the train set
    #        num_classes: number of classes
    #        val_split_amt: how much to split into val from the dataset (will take 1/split_amt). Ignore if None
    #        unlabeled_split_amt: how much to split into unlabeled from the dataset (will take 1/split_amt). Ignore if None
    #
    # Return:
    #        Dict with train, unlabeled, val indices. Field will be None if the split_amt was None
    
    N = len(train_targets)
    
    if val_split_amt is not None:
        val_indices, train_super_indices = _split_dataset(train_targets, num_classes, split_amt=val_split_amt)
    else:
        val_indices = None
        train_super_indices = torch.arange(N)
        
    if unlabeled_split_amt is not None:
        unlabeled_sub_indices, train_sub_indices = _split_dataset(train_targets[train_super_indices],
                                                                  num_classes,
                                                                  split_amt=unlabeled_split_amt)
        unlabeled_indices = train_super_indices[unlabeled_sub_indices]
        train_indices = train_super_indices[train_sub_indices]
    else:
        unlabeled_indices = None
        train_indices = train_super_indices
        
    # Sanity check
    total = len(train_indices)
    if val_indices is not None:
        total += len(val_indices)
    if unlabeled_indices is not None:
        total += len(unlabeled_indices)
    assert total == N
        
    return {
        'val_indices': val_indices,
        'unlabeled_indices': unlabeled_indices,
        'train_indices': train_indices
    }