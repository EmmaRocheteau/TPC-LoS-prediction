import torch
from eICU_preprocessing.reader import eICUReader
from eICU_preprocessing.split_train_test import create_folder
import numpy as np
from models.metrics import print_metrics_regression
from trixi.experiment.pytorchexperiment import PytorchExperiment
import os
from models.shuffle_train import shuffle_train
from eICU_preprocessing.run_all_preprocessing import eICU_path

# view the results by running: python3 -m trixi.browser --port 8080 BASEDIR

def save_to_csv(PyTorchExperimentLogger, data, path, header=None):
    """
        Saves a numpy array to csv in the experiment save dir

        Args:
            data: The array to be stored as a save file
            path: sub path in the save folder (or simply filename)
    """

    folder_path = create_folder(PyTorchExperimentLogger.save_dir, os.path.dirname(path))
    file_path = folder_path + '/' + os.path.basename(path)
    if not file_path.endswith('.csv'):
        file_path += '.csv'
    np.savetxt(file_path, data, delimiter=',', header=header, comments='')
    return

def remove_padding(y, mask, device):
    """
        Filters out padding from tensor of predictions or labels

        Args:
            y: tensor of los predictions or labels
            mask (bool_type): tensor showing which values are padding (0) and which are data (1)
    """
    # note it's fine to call .cpu() on a tensor already on the cpu
    y = y.where(mask, torch.tensor(float('nan')).to(device=device)).flatten().detach().cpu().numpy()
    y = y[~np.isnan(y)]
    return y


class ExperimentTemplate(PytorchExperiment):

    def setup_template(self):

        self.elog.print("Config:")
        self.elog.print(self.config)
        if not self.config.disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # set bool type for where statements
        self.bool_type = torch.cuda.BoolTensor if self.device == torch.device('cuda') else torch.BoolTensor

        self.train_datareader = eICUReader(eICU_path + 'train', device=self.device,
                                           labs_only=self.config.labs_only, no_labs=self.config.no_labs)
        self.val_datareader = eICUReader(eICU_path + 'val', device=self.device,
                                         labs_only=self.config.labs_only, no_labs=self.config.no_labs)
        self.test_datareader = eICUReader(eICU_path + 'test', device=self.device,
                                          labs_only=self.config.labs_only, no_labs=self.config.no_labs)
        self.no_train_batches = len(self.train_datareader.patients) / self.config.batch_size
        self.checkpoint_counter = 0

        self.model = None
        self.optimiser = None

        # add a new function to elog (will save to csv, rather than as a numpy array like elog.save_numpy_data)
        self.elog.save_to_csv = lambda data, filepath, header: save_to_csv(self.elog, data, filepath, header)
        self.remove_padding = lambda y, mask: remove_padding(y, mask, device=self.device)
        self.elog.print('Experiment set up.')

        return

    def train(self, epoch):

        self.model.train()
        if epoch > 0 and self.config.shuffle_train:
            shuffle_train(self.config.eICU_path + 'train')  # shuffle the order of the training data to make the batches different, this takes a bit of time
        train_batches = self.train_datareader.batch_gen(batch_size=self.config.batch_size)
        train_loss = []
        train_y_hat = np.array([])
        train_y = np.array([])

        for batch_idx, (padded, mask, diagnoses, flat, labels, seq_lengths) in enumerate(train_batches):

            self.optimiser.zero_grad()
            y_hat = self.model(padded, diagnoses, flat)
            loss = self.model.loss(y_hat, labels, mask, seq_lengths, self.device, self.config.sum_losses, self.config.loss)
            loss.backward()
            self.optimiser.step()
            train_loss.append(loss.item())

            train_y_hat = np.append(train_y_hat, self.remove_padding(y_hat, mask.type(self.bool_type)))
            train_y = np.append(train_y, self.remove_padding(labels, mask.type(self.bool_type)))

            if self.config.intermediate_reporting and batch_idx % self.config.log_interval == 0 and batch_idx != 0:

                mean_loss_report = sum(train_loss[(batch_idx - self.config.log_interval):-1]) / self.config.log_interval
                self.add_result(value=mean_loss_report,
                                name='Intermediate_Train_Loss',
                                counter=epoch + batch_idx / self.no_train_batches)  # check this
                self.elog.print('Epoch: {} [{:5d}/{:5d} samples] | train loss: {:3.4f}'
                                    .format(epoch,
                                            batch_idx * self.config.batch_size,
                                            batch_idx * self.no_train_batches,
                                            mean_loss_report))
                self.checkpoint_counter += 1

        if not self.config.intermediate_reporting and self.config.mode == 'train':

            print('Train Metrics:')
            mean_train_loss = sum(train_loss) / len(train_loss)
            metrics_list = print_metrics_regression(train_y, train_y_hat, elog=self.elog) # order: mad, mse, mape, msle, r2, kappa
            for metric_name, metric in zip(['mad', 'mse', 'mape', 'msle', 'r2', 'kappa'], metrics_list):
                self.add_result(value=metric, name='train_' + metric_name, counter=epoch)
            self.elog.print('Epoch: {} | Train Loss: {:3.4f}'.format(epoch, mean_train_loss))

        if self.config.mode == 'test':
            print('Done epoch {}'.format(epoch))

        if epoch == self.n_epochs - 1:
            if self.mode == 'train':
                self.save_checkpoint(name='checkpoint', n_iter=epoch)
            if self.config.save_results_csv:
                self.elog.save_to_csv(np.vstack((train_y_hat, train_y)).transpose(),
                                      'train_predictions/epoch{}.csv'.format(epoch),
                                      header='predictions, label')

        return

    def validate(self, epoch):

        if self.config.mode == 'train':
            self.model.eval()
            #if self.config.train_as_val:
            #    val_batches = self.train_datareader.batch_gen(batch_size=self.config.batch_size)
            #else:
            #    val_batches = self.val_datareader.batch_gen(batch_size=self.config.batch_size_test)
            val_batches = self.val_datareader.batch_gen(batch_size=self.config.batch_size_test)
            val_loss = []
            val_y_hat = np.array([])
            val_y = np.array([])

            for (padded, mask, diagnoses, flat, labels, seq_lengths) in val_batches:

                y_hat = self.model(padded, diagnoses, flat)
                loss = self.model.loss(y_hat, labels, mask, seq_lengths, self.device, self.config.sum_losses, self.config.loss)
                val_loss.append(loss.item())  # can't add the model.loss directly because it causes a memory leak

                val_y_hat = np.append(val_y_hat, self.remove_padding(y_hat, mask.type(self.bool_type)))
                val_y = np.append(val_y, self.remove_padding(labels, mask.type(self.bool_type)))

            print('Validation Metrics:')
            mean_val_loss = sum(val_loss) / len(val_loss)
            metrics_list = print_metrics_regression(val_y, val_y_hat, elog=self.elog)  # order: mad, mse, mape, msle, r2, kappa
            for metric_name, metric in zip(['mad', 'mse', 'mape', 'msle', 'r2', 'kappa'], metrics_list):
                self.add_result(value=metric, name='val_' + metric_name, counter=epoch)
            self.elog.print('Epoch: {} | Validation Loss: {:3.4f}'.format(epoch, mean_val_loss))

        elif self.config.mode == 'test' and epoch == self.n_epochs - 1:
            self.test()

        if epoch == self.n_epochs - 1 and self.config.save_results_csv:
            self.elog.save_to_csv(np.vstack((val_y_hat, val_y)).transpose(),
                                  'val_predictions/epoch{}.csv'.format(epoch),
                                  header='predictions, label')

        return

    def test(self):

        self.model.eval()
        test_batches = self.test_datareader.batch_gen(batch_size=self.config.batch_size_test)
        test_loss = []
        test_y_hat = np.array([])
        test_y = np.array([])

        for (padded, mask, diagnoses, flat, labels, seq_lengths) in test_batches:

            y_hat = self.model(padded, diagnoses, flat)
            loss = self.model.loss(y_hat, labels, mask, seq_lengths, self.device, self.config.sum_losses, self.config.loss)
            test_loss.append(loss.item())  # can't add the model.loss directly because it causes a memory leak

            test_y_hat = np.append(test_y_hat, self.remove_padding(y_hat, mask.type(self.bool_type)))
            test_y = np.append(test_y, self.remove_padding(labels, mask.type(self.bool_type)))

        print('Test Metrics:')
        mean_test_loss = sum(test_loss) / len(test_loss)
        metrics_list = print_metrics_regression(test_y, test_y_hat, elog=self.elog)  # order: mad, mse, mape, msle, r2, kappa
        self.elog.save_to_csv(np.vstack((test_y_hat, test_y)).transpose(),
                              'test_predictions.csv',
                              header='predictions, label')
        for metric_name, metric in zip(['mad', 'mse', 'mape', 'msle', 'r2', 'kappa'], metrics_list):
            self.add_result(value=metric, name='test_' + metric_name)
        self.elog.print('Test Loss: {:3.4f}'.format(mean_test_loss))

        with open(self.config.base_dir + '/results.csv', 'a') as f:
            values = self.elog.plot_logger.values
            mad = values['test_mad']['test_mad'][-1][0]
            mse = values['test_mse']['test_mse'][-1][0]
            mape = values['test_mape']['test_mape'][-1][0]
            msle = values['test_msle']['test_msle'][-1][0]
            r2 = values['test_r2']['test_r2'][-1][0]
            kappa = values['test_kappa']['test_kappa'][-1][0]
            f.write('\n{},{},{},{},{},{}'.format(mad, mse, mape, msle, r2, kappa))

        return

    def resume(self, epoch):
        return
