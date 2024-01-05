

import os
from os.path import join, isdir
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
import collections.abc
import argparse
from os.path import join, exists
from multiviewae import me_mVAE, JMVAE
from torchvision.datasets import MNIST
import torch

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

parser.add_argument("k", type=int)
args = parser.parse_args()
config_path = './test_model_configs'

seed_list = [0, 1, 2, 3 , 4]
input_dim = [784, 10]
max_epochs = 500
batch_size = 128

torchvision_dataset = MNIST(root='./bMNIST', train="train", download=False)

images = torchvision_dataset.data.div(255).to(torch.float32)
images = images.view(images.shape[0], -1)
labels = torchvision_dataset.targets
labels_one_hot = torch.zeros(len(labels), 10)
labels_one_hot = labels_one_hot.scatter(1, labels.unsqueeze(1), 1)
labels_one_hot = labels_one_hot

for i, seed in enumerate(seed_list):
    if i == args.k - 1:
        print('training models in fold: {0} with seed: {1}'.format(i, seed))

        dir = './results/CV/kfold_{0}/JMVAE'.format(i)
        in_dict = {'out_dir': dir, 
                'model': {'seed_everything': True, 'seed': seed}}
        
        input_config = createconfig(join(config_path, 'JMVAE.yaml'), in_dict)

        mvae = JMVAE(cfg=input_config,
                     input_dim=input_dim,
                    )
        if not exists(join(dir, "model.ckpt")):
            print('fit JMVAE seed: {0}'.format(seed))
            mvae.fit(images, labels_one_hot, max_epochs=max_epochs, batch_size=batch_size)
        else:
            print('model already trained! skipping...')
            
        dir = './results/CV/kfold_{0}/me_VAE'.format(i)
        in_dict = {'out_dir': dir, 
                'model': {'seed_everything': True, 'seed': seed}}
        
        input_config = createconfig(join(config_path, 'MVAE.yaml'), in_dict)

        mvae = me_mVAE(cfg=input_config,
                    input_dim=input_dim,
                    )
        if not exists(join(dir, "model.ckpt")):
            print('fit mVAE seed: {0}'.format(seed))
            mvae.fit(images, labels_one_hot, max_epochs=max_epochs, batch_size=batch_size)
        else:
            print('model already trained! skipping...')
