from eICU_preprocessing.split_train_test import create_folder
from models.run_lstm import BaselineLSTM
import numpy as np
import random
from models.initialise_arguments import initialise_lstm_arguments


if __name__=='__main__':

    c = initialise_lstm_arguments()
    c['mode'] = 'train'
    c['exp_name'] = 'StandardLSTM'

    # hyper-parameter grid
    param_grid = {
        'n_layers': [1, 2, 3, 4],
        'learning_rate': list(np.logspace(np.log10(0.0001), np.log10(0.01), base=10, num=100)),
        'batch_size': list(int(x) for x in np.logspace(np.log2(4), np.log2(512), base=2, num=8)),
        'lstm_dropout_rate': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'hidden_size': list(int(x) for x in np.logspace(np.log2(16), np.log2(256), base=2, num=5)),
    }

    c['n_layers'] = random.choice(param_grid['n_layers'])
    c['learning_rate'] = round(random.choice(param_grid['learning_rate']), 5)
    c['batch_size'] = random.choice(param_grid['batch_size'])
    c['lstm_dropout_rate'] = random.choice(param_grid['lstm_dropout_rate'])
    c['hidden_size'] = random.choice(param_grid['hidden_size'])

    log_folder_path = create_folder('models/experiments/hyperparameters', c.exp_name)
    standard_lstm = BaselineLSTM(config=c,
                                 n_epochs=c.n_epochs,
                                 name=c.exp_name,
                                 base_dir=log_folder_path,
                                 explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    standard_lstm.run()