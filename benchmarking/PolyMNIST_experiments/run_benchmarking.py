

import os
from os.path import join, isdir
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
import collections.abc
import argparse
from os import listdir
from os.path import isfile, join, exists
from multiviewae import me_mVAE, mmVAE, mmJSD, MoPoEVAE, mvtCAE, mVAE, mmVAEPlus

def update_dict(d, u, l):
    for k, v in u.items():
        if k in l:
            if isinstance(v, collections.abc.Mapping):
                d[k] = update_dict(d.get(k, {}), v, l=v.keys())
            else:
                d[k] = v
    return d

def create_folder(dir_path):
    check_folder = isdir(dir_path)
    if not check_folder:
        os.makedirs(dir_path) 

def updateconfig(orig, update):
    CONFIG_KEYS = [
            "model",
            "datamodule",
            "encoder",
            "decoder",
            "trainer",
            "callbacks",
            "logger",
            "out_dir"
        ]
    OmegaConf.set_struct(orig, True)
    with open_dict(orig):
        # update default cfg with user config
        if update is not None:
            update_keys = list(set(update.keys()) & set(CONFIG_KEYS))
            orig = update_dict(orig, update, l=update_keys)

    if update is not None and update.get('out_dir'):
        orig['out_dir'] = update['out_dir']
    return orig

def createconfig(config_path, in_dict):
    with initialize_config_dir(version_base=None, config_dir=os.getcwd()):
        user_cfg = compose(
                        config_name=config_path,
                        return_hydra_config=True
                    )
    new_cfg = {}
    for key, value in user_cfg.items():
        if str(key) != 'hydra':
            new_cfg[key] = value
    
    new_cfg = OmegaConf.create(new_cfg)
    new_cfg = updateconfig(new_cfg, in_dict)

    create_folder(new_cfg['out_dir'])
    with open(join(new_cfg['out_dir'], 'config.yaml'), "w") as f:
        f.write("# @package _global_\n")
        OmegaConf.save(new_cfg, f)

    input_config = join(new_cfg['out_dir'],'config.yaml')
    return input_config

#load command line args
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#add k as command line argument which has to be an int
parser.add_argument("k", type=int)
args = parser.parse_args()
config_path = './test_model_configs'
train_path = '/path/to/train/data'
train_files = [f for f in listdir(train_path) if isfile(join(train_path, f))]
train_labels = [int(srg[-5]) for srg in train_files]

seed_list = [0, 1, 2, 3 , 4]
input_dim=[(3, 28, 28), (3, 28, 28), (3, 28, 28), (3, 28, 28), (3, 28, 28)]
max_epochs = 300
batch_size = 256
for i, seed in enumerate(seed_list):
    if i == args.k - 1:
        print('training models in fold: {0} with seed: {1}'.format(i, seed))
        dir = './results/CV/kfold_{0}/mVAE'.format(i)
        in_dict = {'out_dir': dir, 
                'model': {'seed_everything': True, 'seed': seed}}
        
        input_config = createconfig(join(config_path, 'MVAE.yaml'), in_dict)

        mvae = mVAE(cfg=input_config,
                    input_dim=input_dim,
                    )
        if not exists(join(dir, "model.ckpt")):
            print('fit mVAE seed: {0}'.format(seed))
            mvae.fit(train_files, labels=train_labels, max_epochs=max_epochs, batch_size=batch_size)
        else:
            print('model already trained! skipping...')

        dir = './results/CV/kfold_{0}/me_mVAE'.format(i)
        in_dict = {'out_dir': dir, 
                'model': {'seed_everything': True, 'seed': seed}}
        
        input_config = createconfig(join(config_path, 'MVAE.yaml'), in_dict)

        mvae = me_mVAE(cfg=input_config,
                    input_dim=input_dim,
                    )
        if not exists(join(dir, "model.ckpt")):
            print('fit mVAE seed: {0}'.format(seed))
            mvae.fit(train_files, labels=train_labels, max_epochs=max_epochs, batch_size=batch_size)
        else:
            print('model already trained! skipping...')
     
        dir = './results/CV/kfold_{0}/mmJSD'.format(i)
        in_dict = {'out_dir': dir, 
                'model': {'seed_everything': True, 'seed': seed}}
        input_config = createconfig(join(config_path, 'mmJSD.yaml'), in_dict)

        mvae = mmJSD(cfg=input_config,
                    input_dim=input_dim,
                    )
        if not exists(join(dir, "model.ckpt")):
            print('fit mmJSD seed: {0}'.format(seed))
            mvae.fit(train_files, labels=train_labels, max_epochs=max_epochs, batch_size=batch_size)
        else:
            print('model already trained! skipping...')


        dir = './results/CV/kfold_{0}/mmVAE'.format(i)
        in_dict = {'out_dir': dir, 
                'model': {'seed_everything': True, 'seed': seed}}
        input_config = createconfig(join(config_path, 'mmVAE.yaml'), in_dict)

        mvae = mmVAE(cfg=input_config,
                    input_dim=input_dim,
                    )
        if not exists(join(dir, "model.ckpt")):
            print('fit mmVAE seed: {0}'.format(seed))
            mvae.fit(train_files, labels=train_labels, max_epochs=max_epochs, batch_size=batch_size)
        else:
            print('model already trained! skipping...')
        
        dir = './results/CV/kfold_{0}/mmVAEPlus'.format(i)

        in_dict = {'out_dir': dir,
                'model': {'seed_everything': True, 'seed': seed}}
        input_config = createconfig(join(config_path, 'mmVAEPlus.yaml'), in_dict)

        mvae = mmVAEPlus(cfg=input_config,
                    input_dim=input_dim,
                    )
        
        if not exists(join(dir, "model.ckpt")):
            print('fit mmVAEPlus seed: {0}'.format(seed))
            mvae.fit(train_files, labels=train_labels, max_epochs=max_epochs, batch_size=batch_size)
        else:
            print('model already trained! skipping...')
            
        dir = './results/CV/kfold_{0}/MoPoEVAE'.format(i)
        in_dict = {'out_dir': dir, 
                'model': {'seed_everything': True, 'seed': seed}}
        input_config = createconfig(join(config_path, 'MoPoEVAE.yaml'), in_dict)

        mvae = MoPoEVAE(cfg=input_config,
                    input_dim=input_dim,
                    )
        
        if not exists(join(dir, "model.ckpt")):
            print('fit MoPoEVAE seed: {0}'.format(seed))
            mvae.fit(train_files, labels=train_labels, max_epochs=max_epochs, batch_size=batch_size)
        else:
            print('model already trained! skipping...')
        
        dir = './results/CV/kfold_{0}/mvtCAE'.format(i)
        in_dict = {'out_dir': dir, 
                'model': {'seed_everything': True, 'seed': seed}}
        input_config = createconfig(join(config_path, 'mvtCAE.yaml'), in_dict)

        mvae = mvtCAE(cfg=input_config,
                    input_dim=input_dim,
                    )
        
        if not exists(join(dir, "model.ckpt")):
            print('fit mvtCAE seed: {0}'.format(seed))
            mvae.fit(train_files, labels=train_labels, max_epochs=max_epochs, batch_size=batch_size)
        else:
            print('model already trained! skipping...')
