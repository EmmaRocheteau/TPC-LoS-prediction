from eICU_preprocessing.split_train_test import create_folder
from models.run_transformer import BaselineTransformer
from models.initialise_arguments import initialise_transformer_arguments
from models.final_experiment_scripts.best_hyperparameters import best_transformer


if __name__=='__main__':

    c = initialise_transformer_arguments()
    c['exp_name'] = 'Transformer12.5'
    c['dataset'] = 'eICU'
    c['percentage_data'] = 12.5
    c = best_transformer(c)

    log_folder_path = create_folder('models/experiments/final/eICU/LoS', c.exp_name)
    transformer = BaselineTransformer(config=c,
                                      n_epochs=c.n_epochs,
                                      name=c.exp_name,
                                      base_dir=log_folder_path,
                                      explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    transformer.run()