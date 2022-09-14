import argparse
import yaml
import pprint
from tqdm import tqdm
import torch
import numpy as np
import os
from torch.cuda.amp import autocast
import torchmetrics

import sys
sys.path.append("..")
from failure_directions.src.config_parsing import ffcv_read_check_override_config
from failure_directions.src.ffcv_utils import get_training_loaders
import failure_directions.src.model_utils as model_utils
import failure_directions.src.trainer as trainer_utils


def evaluate_bce_model(loaded_model, loader, num_classes, set_device=False):
    sigmoid = torch.nn.Sigmoid()
    metric_dict = {}
    with torch.no_grad():
        auc_meter = torchmetrics.AUROC(compute_on_step=False, 
                                     average='macro',
                                     num_classes=num_classes)
        class_auc_meters =  [torchmetrics.AUROC(compute_on_step=False, average='macro', num_classes=1)
                             for _ in range(num_classes)]
        with autocast():
            gts, preds = [], []
            for batch in tqdm(loader):
                x, y, meta, weight = trainer_utils.unwrap_batch(batch, set_device=set_device)
                logits = loaded_model(x)
                gts.append(y.cpu())
                logits = sigmoid(logits)
                preds.append(logits.cpu())
                auc_meter.update(logits.cpu(), y.int().cpu())
                for idx in range(hparams['num_classes']):
                    class_auc_meters[idx].update(logits[:, idx].cpu(), y[:, idx].int().cpu())
    return {
        'classes': torch.cat(gts),
        'preds': torch.cat(preds),
        'AUROC': auc_meter.compute(),
        'Per Class AUROC': [m.compute() for m in class_auc_meters],
    }

def evaluate_ce_model(loaded_model, loader, num_classes, set_device=False):
    with torch.no_grad():
        acc_meter = trainer_utils.AverageMeter()
        class_acc_meters = [trainer_utils.AverageMeter() for _ in range(hparams['num_classes'])]
        with autocast():
            gts, preds = [], []
            for batch in tqdm(loader):
                x, y, meta, weight = trainer_utils.unwrap_batch(batch, set_device=set_device) 
                logits = loaded_model(x)
                gts.append(y.cpu())
                preds.append(logits.argmax(-1).cpu())
                acc_meter.update((logits.argmax(-1) == y).float().mean().item(), len(x))
                for idx in range(hparams['num_classes']):
                    n = (y==idx).float().sum().item()
                    if n > 0:
                        class_acc = ((logits.argmax(-1) == y)[y == idx]).float().mean().item()
                        class_acc_meters[idx].update(class_acc, n)

        return {
            'classes': torch.cat(gts),
            'preds': torch.cat(preds),
            'Accuracy': acc_meter.calculate(),
            'Per Class Accuracy': [m.calculate() for m in class_acc_meters]
        }
        all_metrics[name] = result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FFCV model')
    parser.add_argument('-c', '--config', required=True, type=str, help='dataset config file')
    parser.add_argument('--exp-name', required=True, type=str, help='experiment name')
    parser.add_argument('--disable-logging', action='store_true')
    parser.add_argument('--indices-file', type=str)
    parser.add_argument('--root', type=str, default='/mnt/cfs/projects/correlated_errors/betons')
    # interventions
    parser.add_argument('--loss-vec-file', type=str,
                        help='torch file with weight (for indices of entire beton dataset)')
    args = parser.parse_args()    
    
    with open(args.config, 'r') as file:
        hparams = yaml.safe_load(file)
    hparams = ffcv_read_check_override_config(hparams)
    pprint.pprint(hparams, indent=4)
    
    if args.indices_file is not None:
        indices_dict = torch.load(args.indices_file)
        for k in indices_dict.keys():
            if indices_dict[k] is not None:
                print(k, len(indices_dict[k]))
    else:
        indices_dict = None
        
    if args.loss_vec_file is not None:
        loss_upweight_vec = torch.load(args.loss_vec_file)
    else:
        loss_upweight_vec = None
        
    # For a pytorch dataset, like Colored MNIST, with part of the training set separated out as "unlabeled"
    if 'unlabeled' in hparams.keys() and hparams['unlabeled']:
        train_loader, val_loader, test_loader, unlabeled_loader = get_training_loaders(hparams, root=args.root,
                                                                 indices_dict=indices_dict, get_unlabeled=True)
    else:
        train_loader, val_loader, test_loader = get_training_loaders(hparams, root=args.root,
                                                                 indices_dict=indices_dict)
    training_args = hparams['training']
    training_args['iters_per_epoch'] = len(train_loader)

    build_fn = model_utils.BUILD_FUNCTIONS[hparams['arch_type']]
    model = build_fn(hparams['arch'], hparams['num_classes'])
    model = model.cuda()
    
    set_device = ('cmnist' in hparams.keys() and hparams['cmnist'])
    trainer = trainer_utils.LightWeightTrainer(training_args=training_args, 
                                               exp_name=args.exp_name, 
                                               enable_logging=not args.disable_logging,
                                               loss_upweight_vec=loss_upweight_vec,
                                               bce=hparams['bce'],
                                               set_device=set_device
                                              )
    print("Logging into", trainer.training_dir)
    trainer.fit(model, train_loader, val_loader)
    
    print("=========== Evaluating ==============")
    if args.disable_logging:
        loaded_model = model.eval()
    else:
        checkpoint_path = os.path.join(trainer.training_dir, 'checkpoints', 'checkpoint_last.pt')
        loaded_model = model_utils.load_model(checkpoint_path, build_fn)
        loaded_model = loaded_model.eval().cuda()
    
    all_metrics = {
        'args': vars(args),
        'hparams': hparams,
    }
    for name, loader in [("val", val_loader), ("test", test_loader)]:
        print(name)
        if hparams['bce']:
            all_metrics[name] = evaluate_bce_model(loaded_model, loader, hparams['num_classes'], set_device=set_device)
        else:
            all_metrics[name] = evaluate_ce_model(loaded_model, loader, hparams['num_classes'], set_device=set_device)

    pprint.pprint(all_metrics)
    if not args.disable_logging:
        torch.save(all_metrics, os.path.join(trainer.training_dir, "metrics.pt"))
        print(os.path.join(trainer.training_dir, "metrics.pt"))
        
    
