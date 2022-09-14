import sys
import os
import yaml
import torch
import argparse
import numpy as np
import pprint
import pickle as pkl
import clip
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
from tqdm import tqdm
import failure_directions.src.svm_utils as svm_utils
from failure_directions.src import model_utils
import failure_directions.src.trainer as trainer_utils
from failure_directions.src.config_parsing import ffcv_read_check_override_config, svm_read_check_override_config
from failure_directions.src.ffcv_utils import get_training_loaders
import failure_directions.src.ffcv_utils as ffcv_utils


class SVMFitter:
    def __init__(self, split_and_search=True, balanced=True, cv=2, do_normalize=True, method='SVM', svm_args={}):
        self.split_and_search = split_and_search
        self.balanced = balanced
        self.cv = cv
        self.clfs = None
        self.pre_process = None
        self.do_normalize = do_normalize
        self.method = method
        self.svm_args = svm_args
        
    def set_preprocess(self, train_latents=None):
        self.pre_process = svm_utils.SVMPreProcessing(do_normalize=self.do_normalize)
        if train_latents is not None:
            print("updating whitening")
            self.pre_process.update_stats(train_latents)
        else:
            print("No whitening")
        
    def fit(self, preds, ys, latents):
        assert self.pre_process is not None, 'run set_preprocess on a training set first'
        latents = self.pre_process(latents).numpy()
        clfs, cv_scores = svm_utils.train_per_class_model(latents=latents, ys=ys, 
                                                        preds=preds, balanced=self.balanced, 
                                                        split_and_search=self.split_and_search,
                                                        cv=self.cv, method=self.method, svm_args=self.svm_args)
        self.clfs = clfs
        return cv_scores
    
    def predict(self, ys, latents, compute_metrics=True, preds=None, aux_info={}, verbose=True):
        assert self.clfs is not None, "must call fit first"
        latents = self.pre_process(latents).numpy()
        #ys = ys.numpy()
        #if preds is not None:
        #    preds = preds.numpy()
        return svm_utils.predict_per_class_model(latents=latents, ys=ys, clfs=self.clfs, 
                                     preds=preds, aux_info=aux_info,
                                     verbose=verbose, compute_metrics=compute_metrics, method=self.method) 
    
    def export(self, filename):
        args = {
            'split_and_search': self.split_and_search,
            'balanced': self.balanced,
            'cv': self.cv,
            'do_normalize': self.do_normalize,
        }
        with open(filename, 'wb') as f:
            pkl.dump({
                'clfs': self.clfs,
                'pre_stats': self.pre_process._export(),
                'args': args}, 
                f
            )
    def import_model(self, filename):
        with open(filename, 'rb') as f:
            out = pkl.load(f)
        self.clfs = out['clfs']
        self.split_and_search=out['args']['split_and_search']
        self.balanced = out['args']['balanced']
        self.cv = out['args']['cv']
        self.do_normalize = out['args']['do_normalize']
        svm_stats = out['pre_stats']
        self.pre_process = svm_utils.SVMPreProcessing(do_normalize=self.do_normalize,
                                                      mean=svm_stats['mean'],
                                                      std=svm_stats['std'])
    def legacy_import_model(self, filename, svm_stats):
        with open(filename, 'rb') as f:
            self.clfs = pkl.load(f)
        self.pre_process = svm_utils.SVMPreProcessing(do_normalize=True,
                                                      mean=svm_stats['mean'],
                                                      std=svm_stats['std'])
        
        
                
class CLIPProcessor:
    def __init__(self, ds_mean=0, ds_std=1, 
                 arch='ViT-B/32', device='cuda'):
        self.clip_model, preprocess = clip.load(arch, device=device)
        self.clip_model = self.clip_model.eval()
        clip_normalize = preprocess.transforms[-1]
        self.preprocess_clip = transforms.Compose(
            [
                ffcv_utils.inv_norm(ds_mean, ds_std),
                transforms.Resize((224, 224)),
                clip_normalize,
            ]
        )
        self.device = device
        
    def evaluate_clip_images(self, dataloader):
        clip_activations = []
        with torch.no_grad():
            with autocast():
                for batch in tqdm(dataloader):
                    x = batch[0]
                    x = x.to(self.device)
                    image_features = self.clip_model.encode_image(self.preprocess_clip(x))
                    clip_activations.append(image_features.cpu())
        out = torch.cat(clip_activations).float()
        return out
    
    def evaluate_clip_captions(self, captions):
        text = clip.tokenize(captions)
        ds = torch.utils.data.TensorDataset(text)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, drop_last=False, shuffle=False)
        clip_activations = []
        with torch.no_grad():
            for batch in tqdm(dl):
                caption = batch[0].cuda()
                text_features = self.clip_model.encode_text(caption)
                clip_activations.append(text_features.cpu())
        return torch.cat(clip_activations).float()
   
    def get_caption_scores(self, captions, reference_caption, svm_fitter, target_c):
        caption_latent = self.evaluate_clip_captions(captions)
        reference_latent = self.evaluate_clip_captions([reference_caption])[0]
        latent = caption_latent - reference_latent
        ys = (torch.ones(len(latent))*target_c).long()
        _, decisions = svm_fitter.predict(ys=ys, latents=latent, compute_metrics=False)
        return decisions, caption_latent