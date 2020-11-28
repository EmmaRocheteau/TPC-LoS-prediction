from eICU_preprocessing.split_train_test import create_folder
from torch.optim import Adam
from models.lstm_model import BaseLSTM
from models.experiment_template import ExperimentTemplate
from models.initialise_arguments import initialise_lstm_arguments
from models.final_experiment_scripts.best_hyperparameters import best_cw_lstm


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

    c = initialise_lstm_arguments()
    c['exp_name'] = 'ChannelwiseLSTMNoLabs'
    c['dataset'] = 'eICU'
    c['no_labs'] = True
    c = best_cw_lstm(c)

    log_folder_path = create_folder('models/experiments/final/eICU/LoS', c.exp_name)
    channelwise_lstm = BaselineLSTM(config=c,
                                    n_epochs=c.n_epochs,
                                    name=c.exp_name,
                                    base_dir=log_folder_path,
                                    explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    channelwise_lstm.run()