from eICU_preprocessing.split_train_test import create_folder
from torch.optim import Adam
from models.lstm_model import BaseLSTM
from trixi.util import Config
import argparse
from models.experiment_template import ExperimentTemplate

class BaselineLSTM(ExperimentTemplate):
    def setup(self):
        self.setup_template()
        self.model = BaseLSTM(config=self.config,
                              F=41,
                              D=self.train_datareader.D,
                              no_flat_features=self.train_datareader.no_flat_features).to(device=self.device)
        self.elog.print(self.model)
        self.optimiser = Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.L2_regularisation)
        return

if __name__=='__main__':

    # not hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('-disable_cuda', action='store_true')
    parser.add_argument('--intermediate_reporting', action='store_true')
    parser.add_argument('--exp_name', default='ChannelwiseLSTMNoLabs', type=str)
    parser.add_argument('--mode', default='test', type=str)
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--batch_size_test', default=512, type=int)
    parser.add_argument('-shuffle_train', action='store_true')
    args = parser.parse_args()

    # prepare config dictionary, add all arguments from args
    c = Config()
    for arg in vars(args):
        c[arg] = getattr(args, arg)

    c['n_layers'] = 2
    c['learning_rate'] = 0.00129
    c['batch_size'] = 512
    c['lstm_dropout_rate'] = 0.2
    c['hidden_size'] = 8
    c['last_linear_size'] = 17
    c['diagnosis_size'] = 64
    c['sum_losses'] = True
    c['batchnorm'] = 'mybatchnorm'
    c['loss'] = 'msle'
    c['main_dropout_rate'] = 0.45
    c['bidirectional'] = False
    c['channelwise'] = True
    c['L2_regularisation'] = 0
    c['labs_only'] = False
    c['no_labs'] = True
    c['no_diag'] = False
    c['no_mask'] = False
    c['no_exp'] = False

    log_folder_path = create_folder('models/experiments/final', c.exp_name)
    channelwise_lstm = BaselineLSTM(config=c,
                                    n_epochs=c.n_epochs,
                                    name=c.exp_name,
                                    base_dir=log_folder_path,
                                    explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    channelwise_lstm.run()