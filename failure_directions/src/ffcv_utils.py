import numpy as np
from numpy import dtype
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms 
from dataclasses import replace
from typing import Callable, Optional, Tuple, List

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State
import os

from failure_directions.src.decoders_and_transforms import IMAGE_DECODERS, IMAGE_AUGS
import failure_directions.src.pytorch_datasets as pytorch_datasets
from failure_directions.src.pytorch_datasets import ColoredMNIST

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

def get_ffcv_loader(batch_size, num_workers, beton_path, ds_mean, ds_std, drop_last, shuffle,
                    img_decoder=SimpleRGBImageDecoder(), normalize=True,
                    indices=None, custom_img_transforms=None, custom_label_transforms=None,
                    pipeline_subset=['image', 'label', 'index'], os_cache=False, quasi_random=True,
                    bce=False, root="/mnt/cfs/projects/correlated_errors/betons",
                   ):
    # ds_mean and ds_std should be in uint8 range [0, 255], and should be lists
    # add spurious to pipeline_subset if there is a spurious attribute that you want to surface
    beton_path = os.path.join(root, beton_path)

    pipelines = {}
    
    # Image Processing
    if 'image' in pipeline_subset:
        if custom_img_transforms is None:
            custom_img_transforms = [] 
        image_pipeline = [
            img_decoder,
            *custom_img_transforms,
            ToTensor(), 
            ToDevice(torch.device('cuda'), non_blocking=True), 
            ToTorchImage(),
            Convert(torch.float16),
        ]
        if normalize:
            image_pipeline.append(transforms.Normalize(ds_mean, ds_std))
        pipelines['image'] = image_pipeline
    else:
        pipelines['image'] = None
    
    # Label Processing
    if 'label' in pipeline_subset:
        if custom_label_transforms is None:
            custom_label_transforms = []
        if bce:
            pipelines['label'] = [
                NDArrayDecoder(),
                *custom_label_transforms, 
                ToTensor(),
                ToDevice('cuda:0', non_blocking=True),
                Convert(torch.float16),
            ]
        else:
            pipelines['label'] = [
                IntDecoder(), 
                *custom_label_transforms, 
                ToTensor(), 
                Squeeze(),
                ToDevice('cuda:0', non_blocking=True)
            ]
    else:
        pipelines['label'] = None
        
    # Spurious Processing
    if 'spurious' in pipeline_subset:
        pipelines['spurious'] = [
            IntDecoder(), 
            ToTensor(), 
            Squeeze(),
            ToDevice('cuda:0', non_blocking=True)
        ]
    else:
        pipelines['spurious'] = None
        
    # Index Processing
    if 'index' in pipeline_subset:
        pipelines['index'] = [
            IntDecoder(), 
            ToTensor(), 
            Squeeze(),
            ToDevice('cuda:0', non_blocking=True)
        ]
    else:
        pipelines['index'] = None
        
    # Order Processing    
    RANDOM_ORDER = OrderOption.QUASI_RANDOM if quasi_random else OrderOption.RANDOM
    if shuffle:
        order = RANDOM_ORDER
    else:
        order = OrderOption.SEQUENTIAL
    return Loader(beton_path,
              batch_size=batch_size,
              num_workers=num_workers,
              order=order,
              os_cache=os_cache,
              indices=indices,
              pipelines=pipelines,
              drop_last=drop_last)

def inv_norm(ds_mean, ds_std):
    if ds_std is None:
        return (lambda x: x)
    # invert normalization (useful for visualizing)    
    return transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [255/x for x in ds_std]),
                                transforms.Normalize(mean = [-x /255 for x in ds_mean],
                                                     std = [ 1., 1., 1. ]),
                               ])

def get_records(hparams, split, indices, pipeline_subset=['image', 'label', 'index'], relative_index=True,
               root="/mnt/cfs/projects/correlated_errors/betons",
               indices_dict=None):
    # this is primarily for visualization
    # get the record, no augmentation
    # If relative_index is true, then the indices are the indices of the loader (so after subsetting)
    # otherwise, the indices are the indices of the underlying dataset (so before subsetting)
    # indices
    if relative_index and split != 'test' and indices_dict is not None: 
        if split == 'train':
            master_index = indices_dict['train_indices']
        elif split == 'val':
            master_index = indices_dict['val_indices']
        elif split == 'unlabeled':
            master_index = indices_dict['unlabeled_indices']
        else:
            assert False
        final_indices_to_take = master_index[indices]
    else:
        final_indices_to_take = indices
    
    # get beton
    if split == 'train':
        beton = hparams['train_beton']
    elif split == 'val':
        beton = hparams['val_beton']
        if beton is None:
            print("Pulling val from train")
            beton = hparams['train_beton']
    elif split == 'test':
        beton = hparams['test_beton']
    elif split == 'unlabeled':
        beton = hparams['unlabeled_beton']
        if beton is None:
            print("pulling unlabeled from train")
            beton = hparams['train_beton']
    assert beton is not None

        
    # get aug
    image_aug = None
    if hparams['val_aug'] is not None:
        image_aug = IMAGE_AUGS[hparams['val_aug']](hparams)        
        
    dl_args = {
        'batch_size': hparams['batch_size'],
        'num_workers': hparams['num_workers'],
        'ds_mean': hparams['mean'],
        'ds_std': hparams['std'],
        'normalize': (hparams['mean'] is not None and hparams['std'] is not None),
        'os_cache': hparams['os_cache'],
        'quasi_random': hparams['quasi_random'],
        'pipeline_subset': pipeline_subset,
        'custom_label_transforms': None,
        'shuffle': False,
        'drop_last': False,
        'img_decoder': IMAGE_DECODERS[hparams['val_img_decoder']](hparams['imgsz']),
        'beton_path': beton,
        'custom_img_transforms': image_aug,
        'indices': final_indices_to_take,
        'pipeline_subset': pipeline_subset,
        'bce': hparams['bce'],
        'root': root,
    }
    
    return get_ffcv_loader(**dl_args)

def get_coloredmnist_loaders(hparams, pipeline_subset=['image', 'label', 'index'], 
                         get_unlabeled=False, indices_dict=None, numpy_seed=0):
    # Get Datasets - FROM NOTEBOOK, REDO

    val_split = hparams['val_split']
    train_noise = hparams['train_noise']
    train_corr = hparams['train_corr']
    val_noise = hparams['val_noise']
    val_corr = hparams['val_corr']

    train_base_ds = pytorch_datasets.generate_binary_mnist_ds(root=hparams['mnist_root'], train=True)
    test_ds = pytorch_datasets.generate_binary_mnist_ds(root=hparams['mnist_root'], train=False)

    if get_unlabeled:
        unlabeled_split = hparams['unlabeled_split']
        indices_split = pytorch_datasets.get_unlabeled_indices(train_base_ds.binary_targets, num_classes=2, 
                                                               fold=unlabeled_split, first_val_split=val_split)
        from_train_loaders = ['train', 'val', 'unlabeled']
    else:
        indices_split = pytorch_datasets.create_val_split(train_targets=train_base_ds.binary_targets, num_classes=2, split_amt=val_split)
        from_train_loaders = ['train', 'val']
    
    base_datasets = {'test': test_ds}
    for k in from_train_loaders:
        base_datasets[k] = torch.utils.data.Subset(train_base_ds, indices_split[f'{k}_indices'])
        
    colored_datasets = {
        'train': ColoredMNIST(base_datasets['train'], p_corr=train_corr, noise=train_noise, numpy_seed=numpy_seed),
        'val': ColoredMNIST(base_datasets['val'], p_corr=val_corr, noise=val_noise, numpy_seed=numpy_seed),
        'test': ColoredMNIST(base_datasets['test'], p_corr=val_corr, noise=val_noise, numpy_seed=numpy_seed),
    }

    if get_unlabeled:
        colored_datasets['unlabeled'] = ColoredMNIST(base_datasets['unlabeled'], p_corr=val_corr, noise=train_noise, numpy_seed=numpy_seed)

    datasets = {k: pytorch_datasets.IndexedDataset(v) for k,v in colored_datasets.items()}

    #print("=========")
    #for k in datasets.keys():
    #    print(k, len(datasets[k]))
              
    data_loaders = {}
    for k, v in datasets.items():
        data_loaders[k] = torch.utils.data.DataLoader(v, batch_size=hparams['batch_size'],
                                                      num_workers=hparams['num_workers'],
                                                      shuffle=hparams['shuffle'])

    if get_unlabeled:
        return data_loaders['train'], data_loaders['val'], data_loaders['test'], data_loaders['unlabeled']
    else:
        return data_loaders['train'], data_loaders['val'], data_loaders['test']
    

def get_training_loaders(hparams, pipeline_subset=['image', 'label', 'index'], 
                         get_unlabeled=False, root="/mnt/cfs/projects/correlated_errors/betons",
                         indices_dict=None):

    if 'cmnist' in hparams.keys() and hparams['cmnist']:
        # not sure if all these args are needed
        return get_coloredmnist_loaders(hparams, pipeline_subset=pipeline_subset, get_unlabeled=get_unlabeled, indices_dict=indices_dict)

    common_args = {
        'batch_size': hparams['batch_size'],
        'num_workers': hparams['num_workers'],
        'ds_mean': hparams['mean'],
        'ds_std': hparams['std'],
        'normalize': (hparams['mean'] is not None and hparams['std'] is not None),
        'os_cache': hparams['os_cache'],
        'quasi_random': hparams['quasi_random'],
        'pipeline_subset': pipeline_subset,
        'custom_label_transforms': None,
        'bce': hparams['bce'],
        'root': root,
    }
    
    imgsz = hparams['imgsz']
    
    # decoders
    train_img_decoder = IMAGE_DECODERS[hparams['train_img_decoder']](imgsz)
    val_img_decoder = IMAGE_DECODERS[hparams['val_img_decoder']](imgsz)
    
    # augmentations
    if hparams['train_aug'] is not None:
        train_image_aug = IMAGE_AUGS[hparams['train_aug']](hparams)
    else:
        train_image_aug = None
    if hparams['val_aug'] is not None:
        val_image_aug = IMAGE_AUGS[hparams['val_aug']](hparams)
    else:
        val_image_aug = None
        
    # indices
    if indices_dict is None: 
        train_indices, val_indices, unlabeled_indices = None, None, None
    else:
        train_indices = indices_dict.get('train_indices')
        val_indices = indices_dict.get('val_indices')
        unlabeled_indices = indices_dict.get('unlabeled_indices')
        
        
    # beton_paths
    train_beton = hparams['train_beton']
    val_beton = hparams['val_beton']
    unlabeled_beton = hparams['unlabeled_beton']
    if val_beton is None:
        print("Pulling val from train")
        assert val_indices is not None
        val_beton = train_beton
    if unlabeled_beton is None and get_unlabeled:
        print("Pulling unlabeled from train")
        assert unlabeled_indices is not None
        unlabeled_beton = train_beton
    test_beton = hparams['test_beton']
    assert train_beton is not None
    assert val_beton is not None
    assert test_beton is not None
    
    # get train loader
    print("getting train loader")
    train_loader = get_ffcv_loader(
        beton_path=train_beton, 
        shuffle=hparams['shuffle'],
        drop_last=hparams['drop_last'],
        img_decoder=train_img_decoder,
        indices=train_indices,
        custom_img_transforms=train_image_aug,
        **common_args
    )
    
    common_test_args = {
        'shuffle': False, 'drop_last': False, 'img_decoder': val_img_decoder,
        'custom_img_transforms': val_image_aug,
    }
    
    # get val loader
    print("getting val loader")
    val_loader = get_ffcv_loader(
        beton_path=val_beton, indices=val_indices,
        **common_args, **common_test_args,
    )
    
    # get unlabeled loader
    if get_unlabeled:
        assert unlabeled_indices is not None
        print("getting unlabeled loader")
        unlabeled_loader = get_ffcv_loader(
            beton_path=unlabeled_beton, indices=unlabeled_indices,
            **common_args, **common_test_args,
        )
    
    print("getting test loader")
    test_loader = get_ffcv_loader(
        beton_path=test_beton, indices=None,
        **common_args, **common_test_args,
    )
    
    if get_unlabeled:
        return train_loader, val_loader, test_loader, unlabeled_loader
    else:
        return train_loader, val_loader, test_loader
    
    
