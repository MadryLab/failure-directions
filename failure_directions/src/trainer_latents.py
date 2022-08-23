import argparse
import yaml
import pprint
from tqdm import tqdm
import torch
import numpy as np
import os
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from failure_directions.src.config_parsing import ffcv_read_check_override_config
from failure_directions.src.model_utils import save_model, load_model, save_model_svm
from failure_directions.src.ffcv_utils import get_training_loaders
from failure_directions.src.optimizers import get_optimizer_and_lr_scheduler



def unwrap_batch(batch, set_device=False):
    x, y, spurious, index = None, None, None, None
    if len(batch) == 3:
        x, y, index = batch
    elif len(batch) == 4:
        x, y, spurious, index = batch
    elif len(batch) == 2:
        x, y = batch
    if set_device:
        x = x.cuda()
        y = y.cuda()
    return x, y, spurious, index

class AverageMeter():
    def __init__(self):
        self.num = 0
        self.tot = 0
        
    def update(self, val, sz):
        self.num += val*sz
        self.tot += sz
    
    def calculate(self):
        return self.num/self.tot
    
class LightWeightTrainer():
    def __init__(self, training_args, exp_name, enable_logging=True, loss_upweight_vec=None, set_device=False):
        self.training_args = training_args
        self.ce_loss_unreduced = nn.CrossEntropyLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss()
        self.enable_logging = enable_logging
        if self.enable_logging:
            new_path = self.make_training_dir(exp_name)
            self.training_dir = new_path
            self.writer = SummaryWriter(new_path)
        else:
            self.training_dir = None
            self.writer = None
        self.loss_upweight_vec = loss_upweight_vec
        self.set_device = set_device
        
    def make_training_dir(self, exp_name):
        path = os.path.join('runs', exp_name)
        os.makedirs(path, exist_ok=True)
        existing_count = -1
        
        for f in os.listdir(path):
            if f.startswith('version_'):
                version_num = f.split('version_')[1]
                if version_num.isdigit() and existing_count < int(version_num):
                    existing_count = int(version_num)
        version_num = existing_count + 1
        new_path = os.path.join(path, f"version_{version_num}")
        print("logging in ", new_path)
        os.makedirs(new_path)
        os.makedirs(os.path.join(new_path, 'checkpoints'))
        return new_path
                    
        
    def get_accuracy(self, logits, target):
        correct = logits.argmax(-1) == target
        return (correct.float().mean())

    def get_opt_scaler_scheduler(self, model):
        opt, scheduler = get_optimizer_and_lr_scheduler(self.training_args, model)
        scaler = GradScaler()
        return opt, scaler, scheduler
    
    def training_step(self, model, batch, return_latent=False):
        x, y, spurious, index = unwrap_batch(batch, self.set_device)
        if return_latent:
            latents = []
            def print_out(m, x_):
                latents.append(x_[0].cpu())
            handle = getattr(model, model._last_layer_str).register_forward_pre_hook(print_out)
        logits = model(x)
        if self.loss_upweight_vec is not None:
            weight = self.loss_upweight_vec[index]
            assert (weight > 0).all().item()
            temp = self.ce_loss_unreduced(logits, y) * weight
        else:
            temp = self.ce_loss_unreduced(logits, y)
        loss = temp.mean()
        acc = self.get_accuracy(logits, y)
        if return_latent:
            # remove handle
            handle.remove()
            return loss, acc, len(x), latents[0]
        else:
            return loss, acc, len(x)
    
    def validation_step(self, model, batch):
        x, y, spurious, index = unwrap_batch(batch, self.set_device)
        logits = model(x)
        loss = self.ce_loss(logits, y)
        acc = self.get_accuracy(logits, y)
        return loss, acc, len(x)
    
    def train_epoch(self, epoch_num, model, train_dataloader, opt, scaler, scheduler):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        with tqdm(train_dataloader) as t:
            t.set_description(f"Train Epoch: {epoch_num}")
            first = True
            for batch in t:
                opt.zero_grad(set_to_none=True)
                with autocast():
                    if first:
                        loss, acc, sz, latent = self.training_step(model, batch, return_latent=True)
                        first = False
                    else:
                        loss, acc, sz = self.training_step(model, batch)
                t.set_postfix({'loss': loss.item(), 'acc': acc.item()})
                loss_meter.update(loss.item(), sz)
                acc_meter.update(acc.item(), sz)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                scheduler.step()
        avg_loss, avg_acc = loss_meter.calculate(), acc_meter.calculate()
        return avg_loss, avg_acc, latent
        
    def val_epoch(self, epoch_num, model, val_dataloader):
        model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        with torch.no_grad():
            with tqdm(val_dataloader) as t:
                t.set_description(f"Val Epoch: {epoch_num}")
                for batch in t:
                    with autocast():
                        loss, acc, sz = self.validation_step(model, batch)
                    t.set_postfix({'loss': loss.item(), 'acc': acc.item()})
                    loss_meter.update(loss.item(), sz)
                    acc_meter.update(acc.item(), sz)
        avg_loss, avg_acc = loss_meter.calculate(), acc_meter.calculate()
        return avg_loss, avg_acc
        
        
        
    def fit(self, model, train_dataloader, val_dataloader):
        epochs = self.training_args['epochs']
        opt, scaler, scheduler = self.get_opt_scaler_scheduler(model)
        best_val_loss = np.inf
        latents = []
        for epoch in range(epochs):
            train_loss, train_acc, latent = self.train_epoch(epoch, model, train_dataloader, opt, scaler, scheduler)
            latents.append(latent)
            val_loss, val_acc = self.val_epoch(epoch, model, val_dataloader)
            curr_lr = scheduler.get_last_lr()[0]
            print(f"LR: {curr_lr}, Train Loss: {train_loss:0.4f}, Train Acc: {train_acc:0.4f}, Val Loss: {val_loss:0.4f}, Val Acc: {val_acc:0.4f}")
            
            # Save Checkpoints
            if self.enable_logging:
                
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Acc/train", train_acc, epoch)
                self.writer.add_scalar("Acc/val", val_acc, epoch)
                self.writer.add_scalar("lr", curr_lr, epoch)
                
                run_metadata = {
                    'training_args': self.training_args, 
                    'epoch': epoch, 
                    'training_metrics': {'loss': train_loss, 'acc': train_acc},
                    'val_metrics': {'loss': val_loss, 'acc': val_acc},
                }
                checkpoint_folder = os.path.join(self.training_dir, 'checkpoints')
                checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint_last.pt')
                save_model(model, checkpoint_path, run_metadata)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint_best.pt')
                    save_model(model, checkpoint_path, run_metadata)
                checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint' + str(epoch) + '.pt')
                save_model(model, checkpoint_path, run_metadata)
                if epoch % 5 == 0: # flush every 5 steps
                    self.writer.flush()
        latent_path = os.path.join(checkpoint_folder, 'latents.pt') 
        #latents = torch.cat(latents).detach().numpy()
        torch.save({'latents': latents}, latent_path)

class SVMTrainer():
    def __init__(self, training_args, exp_name, C=1.0, enable_logging=True, set_device=False):
        self.training_args = training_args
        self.C = C

        self.enable_logging = enable_logging
        if self.enable_logging:
            new_path = self.make_training_dir(exp_name)
            self.training_dir = new_path
            self.writer = SummaryWriter(new_path)
        else:
            self.training_dir = None
            self.writer = None
        self.set_device = set_device
        
    def make_training_dir(self, exp_name):
        path = os.path.join('runs', exp_name)
        os.makedirs(path, exist_ok=True)
        existing_count = -1
        
        for f in os.listdir(path):
            if f.startswith('version_'):
                version_num = f.split('version_')[1]
                if version_num.isdigit() and existing_count < int(version_num):
                    existing_count = int(version_num)
        version_num = existing_count + 1
        new_path = os.path.join(path, f"version_{version_num}")
        print("logging in ", new_path)
        os.makedirs(new_path)
        os.makedirs(os.path.join(new_path, 'checkpoints'))
        return new_path
                    
        
    def get_accuracy(self, output, target):
        correct = ((torch.sign(output) + 1) / 2).int() == target
        return (correct.float().mean())

    def get_opt_scaler_scheduler(self, model):
        opt, scheduler = get_optimizer_and_lr_scheduler(self.training_args, model)
        scaler = GradScaler()
        return opt, scaler, scheduler
    
    def training_step(self, model, batch):
        # CHANGE FOR SVM
        # Have it return weights as well? or index of weights in some larger self.loss_upweight_vec?

        #x, y, spurious, index = unwrap_batch(batch, self.set_device)

        x, y, batch_weights = batch['latent'], batch['label'], batch['weight']

        output = model(x)
        # square or not?? 
        loss = torch.mean(batch_weights * torch.clamp(1 - y * output, min=0).pow(2))
        # normalize by number of latents so that the values of C don't depend on that
        loss = loss + (1/self.C)*(model.__dict__['_modules']['linear'].weight.pow(2).mean())

        acc = self.get_accuracy(output, y)

        return loss, acc, len(x)
    
    def train_epoch(self, epoch_num, model, train_dataloader, opt, scaler, scheduler):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        with tqdm(train_dataloader) as t:
            t.set_description(f"Train Epoch: {epoch_num}")
            for batch in t:
                opt.zero_grad(set_to_none=True)
                #with autocast():
                loss, acc, sz = self.training_step(model, batch)
                t.set_postfix({'loss': loss.item(), 'acc': acc.item()})
                loss_meter.update(loss.item(), sz)
                acc_meter.update(acc.item(), sz)

                loss.backward()
                opt.step()

                #scaler.scale(loss).backward()
                #scaler.step(opt)
                #scaler.update()
                scheduler.step()
        avg_loss, avg_acc = loss_meter.calculate(), acc_meter.calculate()
        return avg_loss, avg_acc
        
        
    def fit(self, model, train_dataloader):
        epochs = self.training_args['epochs']
        opt, scaler, scheduler = self.get_opt_scaler_scheduler(model)
        best_val_loss = np.inf
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch, model, train_dataloader, opt, scaler, scheduler)
            #val_loss, val_acc = self.val_epoch(epoch, model, val_dataloader)
            curr_lr = scheduler.get_last_lr()[0]
            print(f"LR: {curr_lr}, Train Loss: {train_loss:0.4f}, Train Acc: {train_acc:0.4f}") #, Val Loss: {val_loss:0.4f}, Val Acc: {val_acc:0.4f}")
            
            # Save Checkpoints
            if self.enable_logging:
                
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                #self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Acc/train", train_acc, epoch)
                #self.writer.add_scalar("Acc/val", val_acc, epoch)
                self.writer.add_scalar("lr", curr_lr, epoch)
                
                run_metadata = {
                    'training_args': self.training_args, 
                    'epoch': epoch, 
                    'training_metrics': {'loss': train_loss, 'acc': train_acc},
                    #'val_metrics': {'loss': val_loss, 'acc': val_acc},
                }
                checkpoint_folder = os.path.join(self.training_dir, 'checkpoints')
                checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint_last.pt')
                save_model_svm(model, checkpoint_path, run_metadata)
                """if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint_best.pt')
                    save_model(model, checkpoint_path, run_metadata)"""
                if epoch % 5 == 0: # flush every 5 steps
                    self.writer.flush()
        return model, checkpoint_path