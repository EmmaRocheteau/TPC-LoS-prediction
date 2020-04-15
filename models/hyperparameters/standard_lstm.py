from eICU_preprocessing.split_train_test import create_folder
from trixi.util import Config
import argparse
from models.run_lstm import BaselineLSTM
import numpy as np
import random

if __name__=='__main__':

    # not hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('-disable_cuda', action='store_true')
    parser.add_argument('--exp_name', default='StandardLSTM', type=str)
    parser.add_argument('-intermediate_reporting', action='store_true')
    parser.add_argument('--n_epochs', default=25, type=int)
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('-shuffle_train', action='store_true')
    parser.add_argument('--mode', default='train', type=str)
    args = parser.parse_args()

    # prepare config dictionary, add all arguments from args
    c = Config()
    for arg in vars(args):
        c[arg] = getattr(args, arg)

    # Hyperparameter grid
    param_grid = {
        'n_layers': [1, 2, 3, 4],
        'learning_rate': list(np.logspace(np.log10(0.0001), np.log10(0.01), base=10, num=100)),
        'batch_size': list(int(x) for x in np.logspace(np.log2(4), np.log2(512), base=2, num=8)),
        'lstm_dropout_rate': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'hidden_size': list(int(x) for x in np.logspace(np.log2(16), np.log2(256), base=2, num=5)),
    }

    c['loss'] = 'msle'
    c['bidirectional'] = False
    c['channelwise'] = False
    c['last_linear_size'] = 17
    c['diagnosis_size'] = 64
    c['batchnorm'] = 'mybatchnorm'
    c['main_dropout_rate'] = 0.45
    c['L2_regularisation'] = 0
    c['n_layers'] = random.choice(param_grid['n_layers'])
    c['learning_rate'] = round(random.choice(param_grid['learning_rate']), 5)
    c['batch_size'] = random.choice(param_grid['batch_size'])
    c['lstm_dropout_rate'] = random.choice(param_grid['lstm_dropout_rate'])
    c['hidden_size'] = random.choice(param_grid['hidden_size'])
    c['sum_losses'] = True
    c['labs_only'] = False
    c['no_labs'] = False
    c['no_diag'] = False
    c['no_mask'] = False
    c['no_exp'] = False

    log_folder_path = create_folder('models/experiments/hyperparameters', c.exp_name)
    standard_lstm = BaselineLSTM(config=c,
                                 n_epochs=c.n_epochs,
                                 name=c.exp_name,
                                 base_dir=log_folder_path,
                                 explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    standard_lstm.run()