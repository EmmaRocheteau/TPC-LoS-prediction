from eICU_preprocessing.split_train_test import create_folder
from trixi.util import Config
import argparse
from models.run_tpc import TPC
import numpy as np
import random

if __name__=='__main__':

    # not hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('-disable_cuda', action='store_true')
    parser.add_argument('--exp_name', default='TempWeightShare', type=str)
    parser.add_argument('--n_epochs', default=25, type=int)
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('--model_type', default='temp_only', type=str)
    parser.add_argument('-shuffle_train', action='store_true')
    parser.add_argument('-intermediate_reporting', action='store_true')
    parser.add_argument('--mode', default='train', type=str)
    args = parser.parse_args()

    # prepare config dictionary, add all arguments from args
    c = Config()
    for arg in vars(args):
        c[arg] = getattr(args, arg)

    c['loss'] = 'msle'
    c['L2_regularisation'] = 0
    c['last_linear_size'] = 17
    c['diagnosis_size'] = 64
    c['batchnorm'] = 'mybatchnorm'
    c['main_dropout_rate'] = 0.45
    c['n_layers'] = 9
    c['kernel_size'] = 4
    c['point_sizes'] = [13]*c['n_layers']  # doesn't actually use this
    c['learning_rate'] = 0.00226
    c['batch_size'] = 32
    c['temp_dropout_rate'] = 0.05
    c['sum_losses'] = True
    c['share_weights'] = True
    c['labs_only'] = False
    c['no_labs'] = False
    c['no_diag'] = False
    c['no_mask'] = False
    c['no_exp'] = False

    temp_kernels_choice = list(int(x) for x in np.logspace(np.log2(16), np.log2(64), base=2, num=9))
    c['temp_kernels'] = [random.choice(temp_kernels_choice)]*c['n_layers']

    log_folder_path = create_folder('models/experiments/hyperparameters', c.exp_name)
    temp_weight_share = TPC(config=c,
                            n_epochs=c.n_epochs,
                            name=c.exp_name,
                            base_dir=log_folder_path,
                            explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    temp_weight_share.run()