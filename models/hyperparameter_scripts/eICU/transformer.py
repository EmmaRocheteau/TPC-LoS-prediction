from eICU_preprocessing.split_train_test import create_folder
from models.run_transformer import BaselineTransformer
import numpy as np
import random
from models.final_experiment_scripts.best_hyperparameters import best_global
from models.initialise_arguments import initialise_transformer_arguments


def get_hyperparam_config(dataset):

    c = initialise_transformer_arguments()
    c['mode'] = 'train'
    c['exp_name'] = 'Transformer'
    if dataset == 'MIMIC':
        c['no_diag'] = True
    c['dataset'] = dataset
    c = best_global(c)

    # hyper-parameter grid
    param_grid = {
        'n_layers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': list(np.logspace(np.log10(0.0001), np.log10(0.01), base=10, num=100)),
        'batch_size': list(int(x) for x in np.logspace(np.log2(4), np.log2(512), base=2, num=8)),
        'trans_dropout_rate': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        'd_model': list(int(x) for x in np.logspace(np.log2(16), np.log2(256), base=2, num=5)),
        'feedforward_size': list(int(x) for x in np.logspace(np.log2(16), np.log2(256), base=2, num=5)),
        'n_heads': [1, 2, 4, 8, 16]
    }

    c['n_layers'] = random.choice(param_grid['n_layers'])
    c['learning_rate'] = round(random.choice(param_grid['learning_rate']), 5)
    c['batch_size'] = random.choice(param_grid['batch_size'])
    c['trans_dropout_rate'] = random.choice(param_grid['trans_dropout_rate'])
    c['d_model'] = random.choice(param_grid['d_model'])
    c['feedforward_size'] = random.choice(param_grid['feedforward_size'])
    c['n_heads'] = random.choice(param_grid['n_heads'])

    return c


if __name__=='__main__':

    for i in range(50):
        try:
            c = get_hyperparam_config('eICU')
            log_folder_path = create_folder('models/experiments/hyperparameters/eICU', c.exp_name)
            transformer = BaselineTransformer(config=c,
                                              n_epochs=c.n_epochs,
                                              name=c.exp_name,
                                              base_dir=log_folder_path,
                                              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
            transformer.run()

        except RuntimeError:
            continue
