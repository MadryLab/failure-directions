import sys
import os
import yaml
import torch
import sys
sys.path.append("..")

import src.svm_utils as svm_utils
import argparse
import numpy as np
from failure_directions.src import model_utils
import failure_directions.src.trainer as trainer_utils
from failure_directions.src.config_parsing import ffcv_read_check_override_config, svm_read_check_override_config
from failure_directions.src.ffcv_utils import get_training_loaders
import pprint
import pickle as pkl
from src.wrappers import SVMFitter, CLIPProcessor


METHODS = ['SVM', 'MLP']

if __name__ == "__main__":
    parser = argparse.ArgumentParser('demo')
    parser.add_argument('-c', '--config', required=True, type=str, help='dataset config file')
    parser.add_argument('-s', '--svm-config', required=True, type=str, help='SVM config file')
    parser.add_argument('--indices-file', type=str)
    parser.add_argument('--root', type=str, default='/mnt/cfs/projects/correlated_errors/betons', help='path to betons')
    parser.add_argument('--model-path', required=True, type=str, help='model path')
    parser.add_argument('--out-file', required=True, type=str, help='output path')
    parser.add_argument('--unlabeled', action='store_true')
    # chest_xray args
    parser.add_argument('--bce', action='store_true', help='bce model')
    parser.add_argument('--bce-class', type=int, default=7)
    parser.add_argument('--spurious', action='store_true', help='If present, include spurious label in the pipeline (assume available in beton)')
    parser.add_argument('--mlp-size', type=int, default=100)
    args = parser.parse_args()
    
    
    # --------- READ THE CONFIGS -------------
    with open(args.config, 'r') as file:
        hparams = yaml.safe_load(file)
    hparams = ffcv_read_check_override_config(hparams)
    hparams['drop_last'] = False
    hparams['shuffle'] = False
    print("=========== Current Config ==================")
    pprint.pprint(hparams, indent=4)
    set_device = ('cmnist' in hparams.keys() and hparams['cmnist'])
    
    with open(args.svm_config, 'r') as file:
        svm_hparams = yaml.safe_load(file)
    svm_hparams = svm_read_check_override_config(svm_hparams)
    print("=========== Current SVM Config ==================")
    pprint.pprint(svm_hparams, indent=4)
    
    if args.spurious:
        pipeline_subset=['image', 'label', 'spurious', 'index']
    else:
        pipeline_subset=['image', 'label', 'index']

    # --------- GET THE LOADERS -------------
    indices_dict = torch.load(args.indices_file) if args.indices_file else None
    common_args = {
        "hparams": hparams,
        "pipeline_subset": pipeline_subset,
        "get_unlabeled": args.unlabeled,
        "root": args.root,
        "indices_dict": indices_dict,
    }
    if args.unlabeled:
        train_loader, val_loader, test_loader, unlabeled_loader = get_training_loaders(**common_args)
        loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader, 'unlabeled': unlabeled_loader}
    else:
        train_loader, val_loader, test_loader = get_training_loaders(**common_args)
        loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    
    # --------- LOAD THE MODEL -------------
    build_fn = model_utils.BUILD_FUNCTIONS[hparams['arch_type']]
    model = model_utils.load_model(args.model_path, build_fn)
    device = torch.device('cuda') if  torch.cuda.is_available() else torch.device('cpu')
    model = model.eval().to(device)    
    if args.bce:
        hparams['num_classes'] = 2
        
    num_classes = hparams['num_classes']
    
    # --------- EVALUATE THE MODEL -------------
    if args.bce:
        # get the BCE threshold
        val_out, bce_threshold = svm_utils.evaluate_bce_loader(loaders['val'], 
                                                               model, 
                                                               class_index=args.bce_class,
                                                               bce_threshold=None)
        out_dict = {'val': val_out}
        for split, loader in loaders.items():
            if split not in out_dict:
                out_dict[split], _ = svm_utils.evaluate_bce_loader(loader,
                                                                  model, 
                                                                  class_index=args.bce_class, 
                                                                  bce_threshold=bce_threshold)
    else:
        out_dict = {
            split: svm_utils.evaluate_loader(loader, model, set_device=set_device) 
            for split, loader in loaders.items()
        }
    metric_dict = {'args': vars(args)}
    if args.bce:
        metric_dict['bce_threshold'] = bce_threshold

    # --------- GET THE EMBEDDINGS -------------
    if svm_hparams['embedding'] == 'inception':
        print("using inception features")
        third_party_dict = {split: svm_utils.evaluate_inception_features(loader) for split, loader in loaders.items()}
    elif svm_hparams['embedding'] == 'clip':
        clip_processor = CLIPProcessor(ds_mean=hparams['mean'], ds_std=hparams['std'], arch='ViT-B/32', device='cuda')
        
        print("using clip features")
        third_party_dict = {
            split: clip_processor.evaluate_clip_images(loader) for split, loader in loaders.items()}
    else:
        third_party_dict = {split: out[split]['latents'] for split in out.keys()}
    
    # --------- PRE PROCESS --------------------
    fit_args = svm_hparams['svm_args']
    fit_args['hidden_layer_size'] = args.mlp_size

    svm_fitter = SVMFitter(split_and_search=fit_args['split_and_search'], balanced=fit_args['balanced'],
                           do_normalize=svm_hparams['normalize'], cv=fit_args['cv'], method=svm_hparams['method'], svm_args=fit_args)
    svm_fitter.set_preprocess(third_party_dict['train'])
    metric_dict['stats'] = svm_fitter.pre_process._export()
    with torch.no_grad():
        third_party_dict = {k: svm_fitter.pre_process(v) for k, v in third_party_dict.items()}
    for k in out_dict.keys():
        out_dict[k]['latents'] = third_party_dict[k]
        
    # --------- Convert to numpy arrays --------
    for k in out_dict.keys():
        for k2 in out_dict[k].keys():
            if torch.is_tensor(out_dict[k][k2]):
                out_dict[k][k2] = out_dict[k][k2].numpy()

    # --------- TRAIN SVM --------------------
    val_out = out_dict['val']
    fit_args = svm_hparams['svm_args']
    cv_scores = svm_fitter.fit(latents=val_out['latents'], ys=val_out['ys'], preds=val_out['preds'])
    metric_dict['cv_scores'] = cv_scores
    for name, ds in out_dict.items():
        print("=================", name, "=====================")
        potential_keys = ['spuriouses', 'indices']

        aux_info = {}
        for k in potential_keys:
            if k  in ds:
                aux_info[k] = ds[k]
        #aux_info = {ds[k] for k in potential_keys if k in ds} # causing an error
        errors, decision, metric = svm_fitter.predict(latents=ds['latents'], 
                                            ys=ds['ys'],
                                            preds=ds['preds'], 
                                            aux_info=aux_info,
                                            compute_metrics=True,
                                            verbose=True,
                                           )
        metric_dict[f"{name}_metrics"] = metric
        metric_dict[f"predicted_{name}_errors"] = 1 - errors # save 0 if correct, 1 otherwise
    svm_fitter.export(f"{args.out_file}_model.pkl")
    torch.save(metric_dict, f"{args.out_file}.pt")