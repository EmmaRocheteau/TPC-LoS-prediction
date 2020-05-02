from eICU_preprocessing.split_train_test import create_folder
from models.run_tpc import TPC
from models.initialise_arguments import initialise_tpc_arguments


if __name__=='__main__':

    c = initialise_tpc_arguments()
    c['mode'] = 'test'
    c['exp_name'] = 'TempWeightShare'
    c['model_type'] = 'temp_only'
    c['share_weights'] = True
    c['temp_kernels'] = [32] * c['n_layers']

    log_folder_path = create_folder('models/experiments/final', c.exp_name)
    temp_weight_share = TPC(config=c,
                            n_epochs=c.n_epochs,
                            name=c.exp_name,
                            base_dir=log_folder_path,
                            explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    temp_weight_share.run()