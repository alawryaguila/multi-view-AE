import torch    
from classifiers import ClfImg as ClfImgCMNIST
import itertools
import os
from os.path import join
from os import listdir
from os.path import isfile, join, exists
from multiviewae import mvtCAE, mmJSD, mmVAE, MoPoEVAE, mmVAEPlus
import pandas as pd
from torchmetrics.classification import MulticlassAccuracy
import numpy as np

def set_clfs(pretrained_classifier_paths, DEVICE):
    clfs = {}
    for m, fp in enumerate(pretrained_classifier_paths):
        model_clf = ClfImgCMNIST()
        model_clf.load_state_dict(torch.load(fp))
        model_clf = model_clf.to(DEVICE)
        clfs["m%d" % m] = model_clf
    return clfs

input_modalities = [0, 1, 2, 3, 4]
#create all possible subsets of input modalities
subsets = []
for i in range(len(input_modalities)):
    subsets.extend(list(itertools.combinations(input_modalities, i+1)))
subsets = subsets[0:-1] #drop case where all modalities are used

#create output modalities ie all modalities not in the input subset
output_modalities = []
for s in subsets:
    output_modalities.append([m for m in input_modalities if m not in s])
print(output_modalities)

#group subsets and output modalities into a dictionary
subset_dict = {}
for i, (s, o) in enumerate(zip(subsets, output_modalities)):
    no_inputs = len(s)
    subset_dict[f'{no_inputs}_inputs_combination_{i}'] = {'input_modalities': s, 'output_modalities': o}
print(subset_dict)

#set up pretrained classifiers
num_modalities = 5
classifier_path = '/path/to/trained_classifiers/trained_clfs_polyMNIST'

test_path = '/path/to/test/data'
test_files = [f for f in listdir(test_path) if isfile(join(test_path, f))]
test_labels = [int(srg[-5]) for srg in test_files]

pretrained_classifier_paths = [f'{classifier_path}/pretrained_img_to_digit_clf_m{m}' for m in range(num_modalities)]
print(pretrained_classifier_paths)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clfs = set_clfs(pretrained_classifier_paths, DEVICE)


#loop over models and k-folds 

input_dims=[(3, 28, 28), (3, 28, 28), (3, 28, 28), (3, 28, 28), (3, 28, 28)]

#create a dataframe that stores results for each model, for each k, for each subset grouping (ie 1 input, 2 input etc)
output_path = './results/CV'
results_df = pd.DataFrame(columns=['model', 'k', 'subset', 'accuracy'])
for k in range(5):
    model_dict = {
            'mmVAEPlus': {'path': f'{output_path}/kfold_{k}/mmVAEPlus', 'model': mmVAEPlus},
            'mmVAE': {'path': f'{output_path}/kfold_{k}/mmVAE', 'model': mmVAE},
            'mmJSD': {'path': f'{output_path}/kfold_{k}/mmJSD', 'model': mmJSD},
            'MTCVAE': {'path': f'{output_path}/CV/kfold_{k}/mvtCAE', 'model': mvtCAE},
            'MoPoE': {'path': f'{output_path}/kfold_{k}/MoPoEVAE', 'model': MoPoEVAE}
            }

    for model_name, model_dict_ in model_dict.items():
        print(model_name)

        model = model_dict_['model'](cfg=f"./test_model_configs/{model_name}.yaml",
                    input_dim=input_dims)
        model = model.load_from_checkpoint(join(model_dict_['path'].format(k=k), 'model.ckpt'))

        #update data_dir 
        model.cfg.datamodule.dataset.data_dir = '/path/to/test/data'
        model.return_mean = False
        if exists(join(model_dict_['path'], 'subset_dict.csv')):
            df = pd.read_csv(join(model_dict_['path'], 'subset_dict.csv'), index_col=0)
        else: 
            #convert nested dictionary to a pandas dataframe
            df = pd.DataFrame.from_dict(subset_dict, orient='index')
            #create a column for accuracy
            df['accuracy'] = 0
            for subset in subset_dict:
                in_modalities = subset_dict[subset]['input_modalities']
                out_modalities = subset_dict[subset]['output_modalities']
                recon = model.predict_reconstruction(test_files, input_modalities=in_modalities, output_modalities=out_modalities, batch_size=256)[0]
                #get prediction accuracy for each modality and average
                acc = 0
                accs_per_class = {m: MulticlassAccuracy(10, average=None).to(DEVICE) for m in out_modalities}
                for i, mod in enumerate(out_modalities):
                    #convert to tensor
                    recon_mod = torch.tensor(recon[i])
                    recon_mod = recon_mod.to(DEVICE)
                    #pass through pretrained classifier
                    clf = clfs[f'm{mod}']
                    clf.eval()
                    with torch.no_grad():
                        pred = clf(recon_mod)
                        acc = accs_per_class[mod](pred, torch.tensor(test_labels).to(DEVICE))

                acc_per_class = {
                        f"{subset}_to_{m}": accs_per_class[m].compute().cpu()
                        for m in accs_per_class
                    }
                acc = {m: acc_per_class[m].mean() for m in acc_per_class}

                #add to df
                mean_pair_acc = np.mean(list(acc.values()))
                df['accuracy'][subset] = mean_pair_acc
            df.to_csv(join(model_dict_['path'], 'subset_dict.csv'))
        #average df accuracy across number of inputs ie 1 input, 2 input etc
        #group by length of input modalities
        df['subset'] = df.index
        
        df['subset'] = df['subset'].str.split('_').str[0]
        df['subset'] = df['subset'].astype(int)
        for subset in df['subset'].unique():
            df_subset = df[df['subset'] == subset]
            mean_acc = df_subset['accuracy'].mean()
            results_curr = pd.DataFrame({'model': model_name, 'k': k, 'subset': subset, 'accuracy': mean_acc}, index=[0])
            results_df = pd.concat([results_df, results_curr], ignore_index=True)

if exists(join(output_path, 'results_df.csv')):
    results_df_old = pd.read_csv(join(output_path, 'results_df.csv'), index_col=0)
    results_df = pd.concat([results_df_old, results_df], ignore_index=True)
results_df.to_csv(join(output_path, 'results_df.csv'))