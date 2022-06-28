def ffcv_read_check_override_config(config, override_args={}):
    # first override args
    for k, v in override_args:
        print(f"Overriding {k}: {config.get(k)} -> {v}")
        config[k] = v
    REQUIRED = ['train_beton', 'test_beton', 'imgsz', 'arch', 'arch_type', 'num_classes']
    for k in REQUIRED:
        assert k in config, f"Missing {k} in config"
    
    DEFAULTS = {
        'batch_size': 512,
        'num_workers': 1,
        'mean': None,
        'std': None,
        'os_cache': False,
        'quasi_random': True,
        'train_img_decoder': 'simple',
        'val_img_decoder': 'simple',
        'train_aug': None,
        'val_aug': None,
        'loss_vec_file': None,
        'shuffle': True,
        'drop_last': True,
        'indices_file': None,
        'val_beton': None,
        'unlabeled_beton': None,
        'loss_upweight': 5,
        'bce': False,
        'cmnist': False,
    }
    for k, v in DEFAULTS.items():
        if k not in config:
            print(f"Using default {k}: {v}")
            config[k] = v
            
    if 'training' not in config:
        config['training'] = {}
        
    TRAINING_DEFAULTS = {
        'epochs': 100,
        'optimizer': {'momentum': 0.9, 'weight_decay': 5e-4},
        'lr': 0.1,
        'lr_scheduler': {'type': 'step_lr', 'step_size': 50, 'gamma': 0.1}
    }
    for k, v in TRAINING_DEFAULTS.items():
        if k not in config['training']:
            print(f"Using default {k}: {v}")
            config['training'][k] = v
    return config

def svm_read_check_override_config(config, override_args={}):
    # first override args
    for k, v in override_args:
        print(f"Overriding {k}: {config.get(k)} -> {v}")
        config[k] = v
    
    DEFAULTS = {
        'method': 'SVM',
        'embedding': 'clip',
        'whiten': True,
        'normalize': False
    }
    for k, v in DEFAULTS.items():
        if k not in config:
            print(f"Using default {k}: {v}")
            config[k] = v
            
    if 'svm_args' not in config:
        config['training'] = {}
        
    TRAINING_DEFAULTS = {
        'split_and_search': True,
        'balanced': True,
        'cv': 2,
    }
    for k, v in TRAINING_DEFAULTS.items():
        if k not in config['svm_args']:
            print(f"Using default {k}: {v}")
            config['svm_args'][k] = v
    return config

