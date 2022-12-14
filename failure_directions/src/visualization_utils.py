import torch
import os
import numpy as np
import torchvision
import yaml
import pprint
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

import failure_directions.src.pytorch_datasets as pytorch_datasets
from failure_directions.src import ffcv_utils
from failure_directions.src import ffcv_utils
from failure_directions.src.config_parsing import ffcv_read_check_override_config
from failure_directions.src.label_maps import CLASS_DICT
import failure_directions.src.model_utils as model_utils
import failure_directions.src.svm_utils as svm_utils
import failure_directions.src.clip_utils as clip_utils
from failure_directions.src.ffcv_utils import get_training_loaders
from failure_directions.src.wrappers import SVMFitter, CLIPProcessor

def convert_to_numpy(d):
    for k in d.keys():
        if torch.is_tensor(d[k]):
            d[k] = d[k].numpy()
        if isinstance(d[k], dict):
            convert_to_numpy(d[k])

class SVMProcessor:
    def __init__(self, svm_filename, root="/mnt/cfs/projects/correlated_errors/betons", get_unlabeled=False,
                 checkpoint_path=None, set_device=False, spurious=False, save_pred_probs=False,
                 batch_size=100
                ):
        self.batch_size = batch_size
        self.get_unlabeled = get_unlabeled
        self.svm_filename = svm_filename
        self.metrics = self._load_svm_result(svm_filename)
        self.spurious = spurious
        convert_to_numpy(self.metrics)
        self.hparams = self._read_hparams_for_viz(self.metrics['args']['config'])
        if self.metrics['args']['indices_file'] is not None:
            self.indices_dict = torch.load(self.metrics['args']['indices_file'])
        else:
            self.indices_dict = None
        self.root = root
#         build model
        model = self._build_model(checkpoint_path=checkpoint_path)
        self.loaders = self._get_loaders()
        
        if self.metrics['args']['bce']:
            class_index = self.metrics['args']['bce_class']
            bce_thresh = self.metrics['bce_threshold']
            self.run_dict = {
                split: svm_utils.evaluate_bce_loader(loader, model, class_index, bce_threshold=bce_thresh, set_device=set_device)[0]
                for split, loader in self.loaders.items()
            }
        else:
            self.run_dict = {
                split: svm_utils.evaluate_loader(loader, model, set_device=set_device, save_pred_probs=save_pred_probs) for split, loader in self.loaders.items()}
        convert_to_numpy(self.run_dict)
        self.orders = self._get_orders()
        self.redo_svm = {}
    
    def _load_svm_result(self, filename):
        metrics = torch.load(filename)
        for k in metrics.keys():
            if 'metrics' not in k:
                continue
            print(f"-----------{k}--------------")
            pprint.pprint({
                "Model Accuracy": metrics[k]['ytrue'].mean(),
                "SVM Accuracy": metrics[k]['accuracy'],
                "SVM Balanced Accuracy": metrics[k]['balanced_accuracy'],
                "Confusion Matrix": metrics[k]['confusion_matrix'],
            })
        return metrics        

    def _read_hparams_for_viz(self, config):
        with open(config, 'r') as file:
            hparams = yaml.safe_load(file)
        hparams = ffcv_read_check_override_config(hparams)
        hparams['batch_size'] = self.batch_size
        hparams['drop_last'] = False
        hparams['shuffle'] = False
        hparams['os_cache'] = False
        hparams['quasi_random'] = True
        print(f"\n-----------CONFIG--------------")
        pprint.pprint(hparams, indent=4)
        return hparams
    
    def _build_model(self, checkpoint_path=None):
        metrics = self.metrics
        hparams = self.hparams
        build_fn = model_utils.BUILD_FUNCTIONS[hparams['arch_type']]
        
        if checkpoint_path is None:
            path = metrics['args']['model_path']
        else:
            path = checkpoint_path
        model = model_utils.load_model(checkpoint_path, build_fn)
        model = model.eval().cuda()
        return model
    
    def _get_loaders(self):
        get_unlabeled = self.get_unlabeled
        if self.spurious:
            pipeline_subset=['image', 'label', 'spurious', 'index']
        else:
            pipeline_subset=['image', 'label', 'index']
        common_args = {
            'hparams': self.hparams, 'get_unlabeled': get_unlabeled, 'root': self.root,
            'indices_dict': self.indices_dict, 'pipeline_subset': pipeline_subset,
        }

        if get_unlabeled:
            train_loader, val_loader, test_loader, unlabeled_loader = ffcv_utils.get_training_loaders(**common_args)
            loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader, 'unlabeled': unlabeled_loader}
        else:
            train_loader, val_loader, test_loader = ffcv_utils.get_training_loaders(**common_args)
            loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
        return loaders
    
    def get_top(self, get_err, target_split, use_confs=False):
        N = self.hparams['num_classes']
        target_metrics = self.metrics[f'{target_split}_metrics']
        all_indices = np.arange(len(target_metrics['classes']))
        if use_confs:
            all_decision_vals = self.run_dict[f'{target_split}']['confs']
        else:
            all_decision_vals = target_metrics['decision_values']
        taken_indices, taken_decision_vals = [], []
        for c in np.arange(N):
            class_mask = (target_metrics['classes'] == c)
            decision_vals = all_decision_vals[class_mask]
            if get_err:
                order = np.argsort(decision_vals)
            else:
                order = np.argsort(decision_vals)[::-1]
            ordered_indices = all_indices[class_mask][order]
            taken_decision_vals.append(decision_vals[order])
            taken_indices.append(ordered_indices)

        return taken_indices, taken_decision_vals

    def _display_images(self, taken_index, taken_scores, taken_confs, split, rows=2, columns=8, filename=None):
        examine_loader = ffcv_utils.get_records(self.hparams, 
                                                split, 
                                                taken_index, 
                                                pipeline_subset=['image', 'label', 'index'], 
                                                relative_index=True,
                                                root=self.root)
        for batch in examine_loader:
            break
        fig, ax = plt.subplots(rows, columns, figsize=(int(columns*1.5), rows*2))
        flat_ax = ax.flatten()
        invTrans = ffcv_utils.inv_norm(self.hparams['mean'], self.hparams['std'])
        for i in range(rows*columns):
            if i >= len(batch[0]):
                break
            flat_ax[i].imshow(torchvision.transforms.ToPILImage()(invTrans(batch[0][i])))
            flat_ax[i].set_title(f"SVM Score: {taken_scores[i].item():0.3f}\n Conf: {taken_confs[i].item():0.3f}", fontsize=9)
            flat_ax[i].axis('off')
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        
    def _get_orders(self):
        orders_to_compute = ['train', 'test', 'val']
        if self.get_unlabeled:
            orders_to_compute.append('unlabeled')
        orders = {}
        for split in orders_to_compute:
            orders[split] = {}
            for name, use_confs in [("CONFIDENCE", True), ("SVM", False)]:
                orders[split][name] = {}
                for get_err in [True, False]: 
                    ind, val = self.get_top(get_err=get_err, target_split=split, use_confs=use_confs)
                    orders[split][name][get_err] = ind
        return orders
        
    def display_extremes(self, c, split, rows=2, columns=8, show_images=True, filename=None):
        run_dict = self.run_dict
        mask = run_dict[split]['ys'] == c
        confs = run_dict[split]['confs']
        dv = self.metrics[f'{split}_metrics']['decision_values']
        sns.scatterplot(x=confs[mask], y=dv[mask])
        plt.show()
        
        if show_images:
            for name, use_confs in [("CONFIDENCE", True), ("SVM", False)]:
                print("----", name, "------")
                for get_err in [True, False]: 
                    fname = None
                    if filename is not None:
                        fname = f"{filename}_{name}_{get_err}.pdf"
                    ind = self.orders[split][name][get_err][c]
                    dv_val = dv[ind]
                    confs_val = confs[ind]
                    self._display_images(taken_index=ind, taken_scores=dv_val, taken_confs=confs_val, split=split, rows=rows, columns=columns, filename=fname)
                    
class ClipAnalyzer:
    def __init__(self, processor, svm_model_name, caption_set_name, class_names, skip_captions=False, legacy_load=True,
                ):
        self.processor = processor
        self.svm_fitter = SVMFitter()
        if legacy_load:
            self.svm_fitter.legacy_import_model(svm_model_name, svm_stats=processor.metrics['stats'])
        else:
            self.svm_fitter.import_model(svm_model_name)
        self.clip_processor = CLIPProcessor(ds_mean=processor.hparams['mean'], 
                                            ds_std=processor.hparams['std'], 
                                            arch='ViT-B/32', device='cuda')
        self.clip_features = {
            split: self.clip_processor.evaluate_clip_images(loader) for split, loader in processor.loaders.items()}
        
        self.test_class = processor.metrics[f'test_metrics']['classes']
        # sanity check loading svm
        mask_ = self.test_class == 0
        lat_ = self.clip_features['test'][mask_]
        ys_ = np.zeros(len(lat_))
        ypred_ = processor.metrics[f'test_metrics']['ypred'][mask_]
        print(
            "consistent with old results",
            (self.svm_fitter.predict(ys=ys_, latents=lat_, compute_metrics=False)[0].astype(int) == ypred_).mean()
        )
        if not skip_captions:
            self.captions = clip_utils.get_caption_set(caption_set_name)
        self.class_names = class_names
        
    def get_svm_style_top_K(self, target_c, caption_type='all', K=10):
        target_c_name = self.class_names[target_c]
        captions = self.captions[target_c_name][caption_type]
        values, caption_latents = self.clip_processor.get_caption_scores(captions=captions, 
                                               reference_caption=self.captions['reference'][target_c],
                                               svm_fitter=self.svm_fitter,
                                               target_c=target_c)
        order = np.argsort(values) # lowest first
        neg_captions = np.array(captions)[order][:K]
        neg_vals = values[order][:K]
        neg_latents = caption_latents.numpy()[order][:K]
        order = order[::-1]
        pos_captions = np.array(captions)[order][:K]
        pos_vals = values[order][:K]
        pos_latents = caption_latents.numpy()[order][:K]
        result = {
            'neg_captions': neg_captions,
            'neg_latents': neg_latents,
            'neg_vals': neg_vals,
            'pos_captions': pos_captions,
            'pos_latents': pos_latents,
            'pos_vals': pos_vals,
        }
        pprint.pprint(result)
        return result
    