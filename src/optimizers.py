from torch.optim import SGD, lr_scheduler
import numpy as np

class LRPolicy():
    def __init__(self, lr_schedule):
        self.lr_schedule = lr_schedule
    def __call__(self, epoch):
        return self.lr_schedule[epoch]

def get_optimizer_and_lr_scheduler(training_params, model):
    iters_per_epoch = training_params['iters_per_epoch']
    optimizer_args = training_params['optimizer']
    lr_scheduler_args = training_params['lr_scheduler']
    epochs = training_params['epochs']
    lr = training_params['lr']
    
    opt = SGD(model.parameters(), 
              lr=lr, 
              momentum=optimizer_args['momentum'],
              weight_decay=optimizer_args['weight_decay'])
    
    scheduler_type = lr_scheduler_args['type']
    
    if scheduler_type == 'constant':
        scheduler = None
    elif scheduler_type == 'cyclic':
        lr_peak_epoch = lr_scheduler_args['lr_peak_epoch']
        lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                        [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                        [0, 1, 0])
        scheduler = lr_scheduler.LambdaLR(opt, LRPolicy(lr_schedule))
#         scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
            
    elif scheduler_type == 'step_lr':
        scheduler = lr_scheduler.StepLR(opt, step_size=lr_scheduler_args['step_size'], 
                                        gamma=lr_scheduler_args['gamma'])
    else:
        raise NotImplementedError("Unimplemented LR Scheduler Type")
    return opt, scheduler