import os
from models.run_lstm import BaselineLSTM
from models.initialise_arguments import initialise_lstm_arguments
from models.final_experiment_scripts.best_hyperparameters import best_lstm


if __name__=='__main__':

    c = initialise_lstm_arguments()
    c['exp_name'] = 'StandardLSTM'
    c['dataset'] = 'eICU'
    c['task'] = 'multitask'
    c = best_lstm(c)

    log_folder_path = 'models/experiments/final/{}/{}/{}'.format(c['dataset'], c['task'], c['exp_name'])
    sub_directories = next(os.walk(log_folder_path))[1]

    for sub_dir in sub_directories:
        baseline_lstm = BaselineLSTM(config=c,
                                     n_epochs=c.n_epochs,
                                     name=c.exp_name,
                                     base_dir=log_folder_path,
                                     explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'},
                                     resume='{}/{}'.format(log_folder_path, sub_dir))
        baseline_lstm.run_test()