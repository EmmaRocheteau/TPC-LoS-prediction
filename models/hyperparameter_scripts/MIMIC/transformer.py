from eICU_preprocessing.split_train_test import create_folder
from models.run_transformer import BaselineTransformer
from models.hyperparameter_scripts.eICU.transformer import get_hyperparam_config


if __name__=='__main__':

    for i in range(50):
        try:
            c = get_hyperparam_config('MIMIC')
            log_folder_path = create_folder('models/experiments/hyperparameters/MIMIC', c.exp_name)
            transformer = BaselineTransformer(config=c,
                                              n_epochs=c.n_epochs,
                                              name=c.exp_name,
                                              base_dir=log_folder_path,
                                              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
            transformer.run()

        except RuntimeError:
            continue