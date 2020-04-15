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
    parser.add_argument('--exp_name', default='TPC', type=str)
    parser.add_argument('--n_epochs', default=25, type=int)
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('--model_type', default='tpc', type=str)
    parser.add_argument('-shuffle_train', action='store_true')
    parser.add_argument('-intermediate_reporting', action='store_true')
    parser.add_argument('--mode', default='train', type=str)
    args = parser.parse_args()

    # prepare config dictionary, add all arguments from args
    c = Config()
    for arg in vars(args):
        c[arg] = getattr(args, arg)

    # Hyperparameter grid
    param_grid = {
        'n_layers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'temp_kernels': list(int(x) for x in np.logspace(np.log2(4), np.log2(16), base=2, num=16)),
        'point_sizes': list(int(x) for x in np.logspace(np.log2(4), np.log2(16), base=2, num=16)),
        'batchnorm': ['mybatchnorm', 'pointonly', 'temponly', 'low_momentum', 'none', 'default'],
        'learning_rate': list(np.logspace(np.log10(0.001), np.log10(0.01), base=10, num=100)),
        'batch_size': list(int(x) for x in np.logspace(np.log2(4), np.log2(512), base=2, num=8)),
        'main_dropout_rate': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        'temp_dropout_rate': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        'last_linear_size': list(int(x) for x in np.logspace(np.log2(16), np.log2(64), base=2, num=16)),
        'diagnosis_size': list(int(x) for x in np.logspace(np.log2(16), np.log2(64), base=2, num=16)),
        'kernel_size': {1: list(range(4, 25)),  # taken out
                        2: [5, 6, 7, 8, 9, 10],
                        3: [3, 4, 5, 6, 7],
                        4: [2, 3, 4, 5, 6],
                        5: [2, 3, 4, 5],
                        6: [2, 3, 4, 5],
                        7: [2, 3, 4, 5],
                        8: [2, 3, 4, 5],
                        9: [3, 4],
                        10: [3, 4],
                        11: [3, 4],
                        12: [3, 4]}
    }

    c['loss'] = 'msle'
    c['L2_regularisation'] = 0
    c['n_layers'] = random.choice(param_grid['n_layers'])
    c['kernel_size'] = random.choice(param_grid['kernel_size'][c['n_layers']])
    c['temp_kernels'] = [random.choice(param_grid['temp_kernels'])]*c['n_layers']
    c['point_sizes'] = [random.choice(param_grid['point_sizes'])]*c['n_layers']
    c['batchnorm'] = random.choice(param_grid['batchnorm'])
    c['learning_rate'] = round(random.choice(param_grid['learning_rate']), 5)
    c['batch_size'] = random.choice(param_grid['batch_size'])
    c['main_dropout_rate'] = random.choice(param_grid['main_dropout_rate'])
    c['temp_dropout_rate'] = random.choice(param_grid['temp_dropout_rate'])
    c['last_linear_size'] = random.choice(param_grid['last_linear_size'])
    c['diagnosis_size'] = random.choice(param_grid['diagnosis_size'])
    c['sum_losses'] = True
    c['share_weights'] = False
    c['labs_only'] = False
    c['no_labs'] = False
    c['no_diag'] = False
    c['no_mask'] = False
    c['no_exp'] = False

    log_folder_path = create_folder('models/experiments/hyperparameters', c.exp_name)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    tpc.run()
