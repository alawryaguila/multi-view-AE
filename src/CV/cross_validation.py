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
from joblib import Parallel, delayed

def corr_coef(brain_data, genetics_data, weights_brain, weights_gen):
    proj_brain = np.dot(brain_data,weights_brain)
    proj_genetics = np.dot(genetics_data,weights_gen)
    corr = np.corrcoef(proj_brain,proj_genetics)
    abs_corr = np.absolute(corr[0,1])
    return abs_corr

class CrossValidation:
    def __init__(self, config, parameters, model_type=None):
        self.config = config
        self.parameters = parameters
        self.model_type = model_type
        self.output_path = self.config['output_dir']

    def gridsearch(self, *data, k_folds=5, jobs=0):

        kfold = KFold(n_splits=k_folds, shuffle=True,random_state=42) #check default split size
        param_combs = list(it.product(*(self.parameters[key] for key in self.parameters)))
        self.input_dims = [np.shape(data_)[1] for data_ in data]
        param_labels_dict = {}
        for i, param_comb in enumerate(param_combs):        
            param_labels_dict['Comb {0}'.format(i)] = {key: param_comb[idx] for idx, key in enumerate(self.parameters)}
        
        if jobs>0:
            print("running array job")
            cv_results = Parallel(n_jobs=jobs)(delayed(self.kfold_cv)(kfold,  param_labels_dict['Comb {0}'.format(i)], *data) for i, param_comb in enumerate(param_combs))
        else:
            cv_results = [self.kfold_cv(kfold, param_labels_dict['Comb {0}'.format(i)], *data) for i, param_comb in enumerate(param_combs)]
        cv_dict = {}
        for i in range(len(param_combs)):        
            cv_dict['Comb {0}'.format(i)] = cv_results[i]
            
        if self.config['sparse']:
            self.model_type = 'sparse_' + self.model_type  
        if self.config['save_model']:
            outpath = self.format_folder()
            json.dump(param_labels_dict,open(join(outpath, 'parameter_labels.json'),'w'))
            json.dump(cv_dict,open(join(outpath, 'cv_results.json'),'w'))
        self.param_labels_dict = param_labels_dict 
        self.cv_dict = cv_dict

    def nested_cv(self, *data, outerfolds=2, innerfolds=2, jobs=0):

        outerfold = KFold(n_splits=outerfolds, shuffle=True,random_state=42) #check default split size
        param_combs = list(it.product(*(self.parameters[key] for key in self.parameters)))
        self.input_dims = [np.shape(data_)[1] for data_ in data]
        print(self.input_dims)
        param_labels_dict = {}
        for i, param_comb in enumerate(param_combs):        
            param_labels_dict['Comb {0}'.format(i)] = {key: param_comb[idx] for idx, key in enumerate(self.parameters)}

        recon_results = np.zeros((2, outerfolds))
        corr_results = []
        dropout_results = []
        self.output_path = self.format_folder()
        for ofold, (train_ids, val_ids) in enumerate(outerfold.split(data[0])):
            print("outer fold: ", ofold)
            train_data, val_data = self.select_data(train_ids, val_ids, *data)
            innerfold = KFold(n_splits=innerfolds, shuffle=True,random_state=42)
            if jobs>0:
                print("running array job")
                cv_results = Parallel(n_jobs=jobs, backend='threading')(delayed(self.kfold_short)(innerfold,  param_labels_dict['Comb {0}'.format(i)], *train_data) for i, param_comb in enumerate(param_combs))
            else:
                cv_results = [self.kfold_short(innerfold, param_labels_dict['Comb {0}'.format(i)], *train_data) for i, param_comb in enumerate(param_combs)]
            indexmin = cv_results.index(min(cv_results))
            bestparams = param_labels_dict['Comb {0}'.format(indexmin)]

            print("Best parameter combination for fold {0}: {1}".format(ofold, bestparams))
            model_outer = self.init_model(self.input_dims, bestparams)
            #print(self.input_dims)
            #print(model_outer.encoders[0])
            model_outer.fit(*train_data)
            same_recon, cross_recon = model_outer.MSE_results(*val_data)
            recon_results[0, ofold], recon_results[1, ofold] = same_recon, cross_recon
            #print(model_outer.encoders[0])
            encoder_weights_brain = model_outer.encoders[0].enc_mean_layer.weight.cpu().detach().numpy()

            encoder_weights_gen = model_outer.encoders[1].enc_mean_layer.weight.cpu().detach().numpy()
            pd.DataFrame(encoder_weights_brain).to_csv(join(self.output_path, 'encoder_weights_brain_fold{0}.csv'.format(ofold)), header=False, index=False)
            pd.DataFrame(train_ids).to_csv(join(self.output_path, 'train_ids_fold{0}.csv'.format(ofold)), header=False, index=False)
            pd.DataFrame(encoder_weights_gen).to_csv(join(self.output_path, 'encoder_weights_gen_fold{0}.csv'.format(ofold)), header=False, index=False)
            dropout = list(model_outer.dropout().cpu().detach().numpy().reshape(-1))
            dropout_results.append(dropout)
            corr_temp = []
            for i in range(np.shape(encoder_weights_brain)[0]):
                corr = corr_coef(val_data[0], val_data[1], encoder_weights_brain[i,:], encoder_weights_gen[i,:])
                corr_temp.append(corr)
            corr_results.append(corr_temp)
            #COME BACK TO - creating separate latent spaces for each encoder (using test set) and correlating
           # corr_temp = []
           # latent_1 = model_outer.
           # for i in range(np.shape(encoder_weights_brain)[0]):
           #     corr = np.corr_coef()
        pd.DataFrame(dropout_results).to_csv(join(self.output_path, 'dropout_results.csv'))
        pd.DataFrame(recon_results).to_csv(join(self.output_path, 'reconstruction_results.csv'))
        pd.DataFrame(corr_results).to_csv(join(self.output_path, 'corr_results.csv'))
        #pd.DataFrame(latent_corr_results).to_csv(join(self.output_path, 'latent_corr_results.csv'))
        return self

    def kfold_short(self, kfold, parameters, *data):
        recon = 0
        folds = 0
        print("parameters: ", parameters)
        for fold, (train_ids, val_ids) in enumerate(kfold.split(data[0])):
            print("inner fold: ", fold)
            folds+=1
            train_data, val_data = self.select_data(train_ids, val_ids, *data)
            model = self.init_model(self.input_dims, parameters)
            #print(self.input_dims)
            model.fit(*train_data)
            same_recon, cross_recon = model.MSE_results(*val_data)
            recon += (same_recon + cross_recon)/2
        recon = recon/folds
        return recon

    def kfold_cv(self, kfold, parameters, *data):
        cv_dict = {}
        for fold, (train_ids, val_ids) in enumerate(kfold.split(data[0])):
            #print("fold: ", fold)
            train_data, val_data = self.select_data(train_ids, val_ids, *data)
            model = self.init_model(self.input_dims, parameters)
            model.fit(*train_data)

            same_recon, cross_recon = model.MSE_results(*val_data)
            cv_dict['fold {0}'.format(fold)] = {'same view MSE': round(same_recon, 4), 'cross view MSE': round(cross_recon, 4)}
        return cv_dict
        
    def init_model(self, input_dims, parameters):
        self.input_dims = input_dims
        use_GPU = self.config['use_GPU']
        use_cuda = use_GPU and torch.cuda.is_available()
        self.kwargs_generator = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        self.device = torch.device("cuda" if use_cuda else "cpu")
        #update config file depending on parameters
        config = self.config
        for key,val in parameters.items():
            config[key] = val
       # print(config)
        if self.model_type=='joint_VAE':
            from ..models.joint_vae import VAE
          #  print(config['latent_size'])
          #  print(config['hidden_layers'])
            return VAE(self.input_dims, config).to(self.device)
        elif self.model_type=='VAE':
            from ..models.vae import VAE
            return VAE(self.input_dims, config).to(self.device)
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
    