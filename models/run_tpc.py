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
                              F=self.train_datareader.F,
                              D=self.train_datareader.D,
                              no_flat_features=self.train_datareader.no_flat_features).to(device=self.device)
        self.elog.print(self.model)
        self.optimiser = Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.L2_regularisation)
        return

if __name__=='__main__':

    # not hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('-disable_cuda', action='store_true')
    parser.add_argument('-intermediate_reporting', action='store_true')
    parser.add_argument('--exp_name', default='TPC', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--n_epochs', default=15, type=int)
    parser.add_argument('--batch_size_test', default=128, type=int)
    parser.add_argument('--model_type', default='tpc', type=str)
    parser.add_argument('--loss', default='msle', type=str, help='can either be msle or mse')
    parser.add_argument('-share_weights', action='store_true')
    parser.add_argument('-shuffle_train', action='store_true')

    # hyperparams
    parser.add_argument('--n_layers', default=9, type=int)
    parser.add_argument('--L2_regularisation', default=0, type=float)
    parser.add_argument('--kernel_size', default=4, type=int)
    parser.add_argument('--no_temp_kernels', default=12, type=int)
    parser.add_argument('--point_size', default=13, type=int)
    parser.add_argument('--last_linear_size', default=17, type=int)
    parser.add_argument('--diagnosis_size', default=64, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=0.00226, type=float)
    parser.add_argument('--main_dropout_rate', default=0.45, type=float)
    parser.add_argument('--temp_dropout_rate', default=0.05, type=float)
    parser.add_argument('-sum_losses', action='store_true')
    parser.add_argument('-labs_only', action='store_true')
    parser.add_argument('-no_mask', action='store_true')
    parser.add_argument('-no_diag', action='store_true')
    parser.add_argument('-no_labs', action='store_true')
    parser.add_argument('-no_exp', action='store_true')
    parser.add_argument('--batchnorm', default='mybatchnorm', type=str, help='can be: none, pointwiseonly, temponly, default, '
                                                                      'mybatchnorm or low_momentum. '
                                                                      '\nfconly, convonly and low_momentum are '
                                                                      'implemented with mybatchnorm rather than default '
                                                                      'pytorch')
    args = parser.parse_args()

    # prepare config dictionary, add all arguments from args
    c = Config()
    for arg in vars(args):
        c[arg] = getattr(args, arg)

    c['temp_kernels'] = [args.no_temp_kernels]*c['n_layers']
    c['point_sizes'] = [args.point_size]*c['n_layers']

    log_folder_path = create_folder('models/experiments', c.exp_name)

    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S'})
    if c.mode == 'train':
        tpc.run()
    if c.mode == 'test':
        tpc.run_test()
