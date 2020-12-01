from eICU_preprocessing.split_train_test import create_folder
from models.run_tpc import TPC
import numpy as np
import random
from models.final_experiment_scripts.best_hyperparameters import best_global
from models.initialise_arguments import initialise_tpc_arguments


def get_hyperparam_config(dataset):

    c = initialise_tpc_arguments()
    c['mode'] = 'train'
    c['exp_name'] = 'TPC'
    if dataset == 'MIMIC':
        c['no_diag'] = True
    c['dataset'] = dataset
    c['model_type'] = 'tpc'
    c = best_global(c)

    # hyper-parameter grid
    param_grid = {
        'n_layers': [5, 6, 7, 8, 9, 10, 11, 12],  # most promising range is 5-12 layers
        'temp_kernels': list(int(x) for x in np.logspace(np.log2(4), np.log2(16), base=2, num=16)),
        'point_sizes': list(int(x) for x in np.logspace(np.log2(4), np.log2(16), base=2, num=16)),
        'learning_rate': list(np.logspace(np.log10(0.001), np.log10(0.01), base=10, num=100)),
        'batch_size': list(int(x) for x in np.logspace(np.log2(4), np.log2(512), base=2, num=8)),  # might have to search between 4 and 32 to avoid memory issues
        'temp_dropout_rate': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        'kernel_size': {#1: list(range(4, 25)),
                        #2: [5, 6, 7, 8, 9, 10],
                        #3: [3, 4, 5, 6, 7],
                        #4: [2, 3, 4, 5, 6],
                        5: [2, 3, 4, 5],
                        6: [2, 3, 4, 5],
                        7: [2, 3, 4, 5],
                        8: [2, 3, 4, 5],
                        9: [3, 4],
                        10: [3, 4],
                        11: [3, 4],
                        12: [3, 4]}
    }

    c['n_layers'] = random.choice(param_grid['n_layers'])
    c['kernel_size'] = random.choice(param_grid['kernel_size'][c['n_layers']])
    c['temp_kernels'] = [random.choice(param_grid['temp_kernels'])] * c['n_layers']
    c['point_sizes'] = [random.choice(param_grid['point_sizes'])] * c['n_layers']
    c['learning_rate'] = round(random.choice(param_grid['learning_rate']), 5)
    c['batch_size'] = random.choice(param_grid['batch_size'])
    c['temp_dropout_rate'] = random.choice(param_grid['temp_dropout_rate'])

    return c


if __name__=='__main__':


    for i in range(50):
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