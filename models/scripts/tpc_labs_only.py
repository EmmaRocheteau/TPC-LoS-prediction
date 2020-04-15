from eICU_preprocessing.split_train_test import create_folder
from torch.optim import Adam
from models.tpc_model import TempPointConv
from trixi.util import Config
import argparse
from models.experiment_template import ExperimentTemplate

class TPC(ExperimentTemplate):
    def setup(self):
        self.setup_template()
        self.model = TempPointConv(config=self.config,
                              F=46,
                              D=self.train_datareader.D,
                              no_flat_features=self.train_datareader.no_flat_features).to(device=self.device)
        self.elog.print(self.model)
        self.optimiser = Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.L2_regularisation)
        return

if __name__=='__main__':

    # not hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('-disable_cuda', action='store_true')
    parser.add_argument('--exp_name', default='TPCLabsOnly', type=str)
    parser.add_argument('--n_epochs', default=15, type=int)
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('--model_type', default='tpc', type=str)
    parser.add_argument('-shuffle_train', action='store_true')
    parser.add_argument('-intermediate_reporting', action='store_true')
    parser.add_argument('--mode', default='test', type=str)
    args = parser.parse_args()

    # prepare config dictionary, add all arguments from args
    c = Config()
    for arg in vars(args):
        c[arg] = getattr(args, arg)

    c['loss'] = 'msle'
    c['diagnosis_size'] = 64
    c['last_linear_size'] = 17
    c['batchnorm'] = 'mybatchnorm'
    c['main_dropout_rate'] = 0.45
    c['L2_regularisation'] = 0
    c['n_layers'] = 9
    c['kernel_size'] = 4
    c['temp_kernels'] = [12]*c['n_layers']
    c['point_sizes'] = [13]*c['n_layers']
    c['learning_rate'] = 0.00226
    c['batch_size'] = 32
    c['temp_dropout_rate'] = 0.05
    c['sum_losses'] = True
    c['share_weights'] = False
    c['labs_only'] = True
    c['no_labs'] = False
    c['no_diag'] = False
    c['no_mask'] = False
    c['no_exp'] = False

    log_folder_path = create_folder('models/experiments/final', c.exp_name)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    tpc.run()