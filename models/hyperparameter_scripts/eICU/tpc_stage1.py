from eICU_preprocessing.split_train_test import create_folder
from models.run_tpc import TPC
import numpy as np
import random
from models.initialise_arguments import initialise_tpc_arguments


def get_hyperparam_config(dataset):

    c = initialise_tpc_arguments()
    c['mode'] = 'train'
    c['exp_name'] = 'TPC'
    if dataset == 'MIMIC':
        c['no_diag'] = True
    c['dataset'] = dataset
    c['model_type'] = 'tpc'

    # hyper-parameter grid
    param_grid = {
        #'batchnorm': ['mybatchnorm', 'pointonly', 'temponly', 'low_momentum', 'none', 'default'],
        'main_dropout_rate': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        'last_linear_size': list(int(x) for x in np.logspace(np.log2(16), np.log2(64), base=2, num=16)),
        'diagnosis_size': list(int(x) for x in np.logspace(np.log2(16), np.log2(64), base=2, num=16)),
    }

    #c['batchnorm'] = random.choice(param_grid['batchnorm'])
    c['main_dropout_rate'] = random.choice(param_grid['main_dropout_rate'])
    c['last_linear_size'] = random.choice(param_grid['last_linear_size'])
    c['diagnosis_size'] = random.choice(param_grid['diagnosis_size'])

    return c


if __name__=='__main__':

    for i in range(25):
        try:
            c = get_hyperparam_config('eICU')
            log_folder_path = create_folder('models/experiments/hyperparameters/eICU', c.exp_name)
            tpc = TPC(config=c,
                      n_epochs=c.n_epochs,
                      name=c.exp_name,
                      base_dir=log_folder_path,
                      explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
            tpc.run()

        except RuntimeError:
            continue
