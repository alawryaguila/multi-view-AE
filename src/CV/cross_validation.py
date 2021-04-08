'''
Class for conducting cross validation

'''
import torch
import itertools as it
import numpy as np 
import pandas as pd 
import os
from os.path import join
import datetime
import json
from sklearn.model_selection import KFold
class CrossValidation:
    def __init__(self, config, parameters, model_type=None):
        self.config = config
        self.parameters = parameters
        self.model_type = model_type
        self.output_path = self.config['output_dir']

    def run_CV(self, *data, k_folds=2):

        kfold = KFold(n_splits=k_folds, shuffle=True,random_state=42) #check default split size
        param_combs = list(it.product(*(self.parameters[key] for key in self.parameters)))
        input_dims = [np.shape(data_)[1] for data_ in data]
        param_labels_dict = {}
        cv_dict = {}
        for i, param_comb in enumerate(param_combs):        
            param_labels_dict['Comb {0}'.format(i)] = {key: param_comb[idx] for idx, key in enumerate(self.parameters)}
            cv_dict['Comb {0}'.format(i)] = {}
            print("parameters: ", param_labels_dict['Comb {0}'.format(i)])
            for fold, (train_ids, val_ids) in enumerate(kfold.split(data[0])):
                print("fold: ", fold)
                train_data, val_data = self.select_data(train_ids, val_ids, *data)
                model = self.init_model(input_dims, param_labels_dict['Comb {0}'.format(i)])
                model.fit(*train_data)
                same_recon, cross_recon = model.MSE_results(*val_data)
                cv_dict['Comb {0}'.format(i)]['fold {0}'.format(fold)] = {'same view MSE': round(same_recon, 4), 'cross view MSE': round(cross_recon, 4)}
        #save validation results
        outpath = self.format_folder()
        json.dump(param_labels_dict,open(join(outpath, 'parameter_labels.json'),'w'))
        json.dump(cv_dict,open(join(outpath, 'cv_results.json'),'w'))
        self.param_labels_dict = param_labels_dict 
        self.cv_dict = cv_dict

    def init_model(self, input_dims, parameters):
        self.input_dims = input_dims
        DEVICE = torch.device("cuda")
        #update config file depending on parameters
        config = self.config
        for key,val in parameters.items():
            config[key] = val
        if self.model_type=='joint_VAE':
            from models.joint_vae import VAE
            return VAE(self.input_dims, config).to(DEVICE)
        elif self.model_type=='VAE':
            from models.vae import VAE
            return VAE(self.input_dims, config).to(DEVICE)
        #TO DO rest of models
        else:
            print("model type not recognised")
            exit()

    def select_data(self, train_ids,val_ids, *data):
        train_data = []
        val_data = []
        for data_ in data:
            train_data.append(data_[train_ids,:])
            val_data.append(data_[val_ids,:])
        return train_data, val_data

    def format_folder(self):
        init_path = self.config['output_dir']
        model_type = self.model_type
        date = str(datetime.date.today())
        date = date.replace('-', '_')
        output_path = init_path + '/' + model_type + '/' + date
        if not os.path.exists(output_path):
            os.makedirs(output_path)   
        return output_path