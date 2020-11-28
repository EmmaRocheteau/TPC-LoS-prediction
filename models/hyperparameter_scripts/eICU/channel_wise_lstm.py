from eICU_preprocessing.split_train_test import create_folder
from models.run_lstm import BaselineLSTM
import numpy as np
import random
from models.final_experiment_scripts.best_hyperparameters import best_lstm
from models.initialise_arguments import initialise_lstm_arguments


# this script should be run after the standard LSTM has been optimised already and the values put into models/final_experiment_scripts/best_hyperparameters


def get_hyperparam_config(dataset):

    c = initialise_lstm_arguments()
    c['mode'] = 'train'
    c['exp_name'] = 'ChannelwiseLSTM'
    if dataset == 'MIMIC':
        c['no_diag'] = True
    c['dataset'] = dataset
    c = best_lstm(c)  # get best hyperparams from standard LSTM for this dataset
    c['channelwise'] = True

    hidden_size_choice = list(int(x) for x in np.logspace(np.log2(4), np.log2(16), base=2, num=6))
    c['hidden_size'] = random.choice(hidden_size_choice)

    return c


if __name__=='__main__':

    for i in range(10):
        try:
            c = get_hyperparam_config('eICU')
            log_folder_path = create_folder('models/experiments/hyperparameters/eICU', c.exp_name)
            channel_wise_lstm = BaselineLSTM(config=c,
                                             n_epochs=c.n_epochs,
                                             name=c.exp_name,
                                             base_dir=log_folder_path,
                                             explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
            channel_wise_lstm.run()

        except RuntimeError:
            continue