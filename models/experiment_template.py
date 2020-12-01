import torch
from eICU_preprocessing.reader import eICUReader
from MIMIC_preprocessing.reader import MIMICReader
from eICU_preprocessing.split_train_test import create_folder
import numpy as np
from models.metrics import print_metrics_regression, print_metrics_mortality
from trixi.experiment.pytorchexperiment import PytorchExperiment
import os
from models.shuffle_train import shuffle_train
from eICU_preprocessing.run_all_preprocessing import eICU_path
from MIMIC_preprocessing.run_all_preprocessing import MIMIC_path


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

        # get datareader
        if self.config.dataset == 'MIMIC':
            self.datareader = MIMICReader
            self.data_path = MIMIC_path
        else:
            self.datareader = eICUReader
            self.data_path = eICU_path
        self.train_datareader = self.datareader(self.data_path + 'train', device=self.device,
                                           labs_only=self.config.labs_only, no_labs=self.config.no_labs)
        self.val_datareader = self.datareader(self.data_path + 'val', device=self.device,
                                         labs_only=self.config.labs_only, no_labs=self.config.no_labs)
        self.test_datareader = self.datareader(self.data_path + 'test', device=self.device,
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

    def train(self, epoch, mort_pred_time=24):

        self.model.train()
        if epoch > 0 and self.config.shuffle_train:
            shuffle_train(self.config.eICU_path + 'train')  # shuffle the order of the training data to make the batches different, this takes a bit of time
        train_batches = self.train_datareader.batch_gen(batch_size=self.config.batch_size)
        train_loss = []
        train_y_hat_los = np.array([])
        train_y_los = np.array([])
        train_y_hat_mort = np.array([])
        train_y_mort = np.array([])

        for batch_idx, batch in enumerate(train_batches):

            if batch_idx > (self.no_train_batches // (100 / self.config.percentage_data)):
                break

            # unpack batch
            if self.config.dataset == 'MIMIC':
                padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
                diagnoses = None
            else:
                padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch

            self.optimiser.zero_grad()
            y_hat_los, y_hat_mort = self.model(padded, diagnoses, flat)
            loss = self.model.loss(y_hat_los, y_hat_mort, los_labels, mort_labels, mask, seq_lengths, self.device,
                                   self.config.sum_losses, self.config.loss)
            loss.backward()
            self.optimiser.step()
            train_loss.append(loss.item())

            if self.config.task in ('LoS', 'multitask'):
                train_y_hat_los = np.append(train_y_hat_los, self.remove_padding(y_hat_los, mask.type(self.bool_type)))
                train_y_los = np.append(train_y_los, self.remove_padding(los_labels, mask.type(self.bool_type)))
            if self.config.task in ('mortality', 'multitask') and mort_labels.shape[1] >= mort_pred_time:
                train_y_hat_mort = np.append(train_y_hat_mort,
                                             self.remove_padding(y_hat_mort[:, mort_pred_time],
                                                                 mask.type(self.bool_type)[:, mort_pred_time]))
                train_y_mort = np.append(train_y_mort, self.remove_padding(mort_labels[:, mort_pred_time],
                                                                           mask.type(self.bool_type)[:, mort_pred_time]))

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
            if self.config.task in ('LoS', 'multitask'):
                los_metrics_list = print_metrics_regression(train_y_los, train_y_hat_los, elog=self.elog) # order: mad, mse, mape, msle, r2, kappa
                for metric_name, metric in zip(['mad', 'mse', 'mape', 'msle', 'r2', 'kappa'], los_metrics_list):
                    self.add_result(value=metric, name='train_' + metric_name, counter=epoch)
            if self.config.task in ('mortality', 'multitask'):
                mort_metrics_list = print_metrics_mortality(train_y_mort, train_y_hat_mort, elog=self.elog)
                for metric_name, metric in zip(['acc', 'prec0', 'prec1', 'rec0', 'rec1', 'auroc', 'auprc', 'f1macro'], mort_metrics_list):
                    self.add_result(value=metric, name='train_' + metric_name, counter=epoch)
            self.elog.print('Epoch: {} | Train Loss: {:3.4f}'.format(epoch, mean_train_loss))

        if self.config.mode == 'test':
            print('Done epoch {}'.format(epoch))

        if epoch == self.n_epochs - 1:
            if self.config.mode == 'train':
                self.save_checkpoint(name='checkpoint', n_iter=epoch)
            if self.config.save_results_csv:
                if self.config.task in ('LoS', 'multitask'):
                    self.elog.save_to_csv(np.vstack((train_y_hat_los, train_y_los)).transpose(),
                                          'train_predictions_los/epoch{}.csv'.format(epoch),
                                          header='los_predictions, label')
                if self.config.task in ('mortality', 'multitask'):
                    self.elog.save_to_csv(np.vstack((train_y_hat_mort, train_y_mort)).transpose(),
                                          'train_predictions_mort/epoch{}.csv'.format(epoch),
                                          header='mort_predictions, label')

        return

    def validate(self, epoch, mort_pred_time=24):

        if self.config.mode == 'train':
            self.model.eval()
            val_batches = self.val_datareader.batch_gen(batch_size=self.config.batch_size_test)
            val_loss = []
            val_y_hat_los = np.array([])
            val_y_los = np.array([])
            val_y_hat_mort = np.array([])
            val_y_mort = np.array([])

            for batch in val_batches:

                # unpack batch
                if self.config.dataset == 'MIMIC':
                    padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
                    diagnoses = None
                else:
                    padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch

                y_hat_los, y_hat_mort = self.model(padded, diagnoses, flat)
                loss = self.model.loss(y_hat_los, y_hat_mort, los_labels, mort_labels, mask, seq_lengths, self.device,
                                       self.config.sum_losses, self.config.loss)
                val_loss.append(loss.item())  # can't add the model.loss directly because it causes a memory leak

                if self.config.task in ('LoS', 'multitask'):
                    val_y_hat_los = np.append(val_y_hat_los,
                                                self.remove_padding(y_hat_los, mask.type(self.bool_type)))
                    val_y_los = np.append(val_y_los, self.remove_padding(los_labels, mask.type(self.bool_type)))
                if self.config.task in ('mortality', 'multitask') and mort_labels.shape[1] >= mort_pred_time:
                    val_y_hat_mort = np.append(val_y_hat_mort,
                                                 self.remove_padding(y_hat_mort[:, mort_pred_time],
                                                                     mask.type(self.bool_type)[:, mort_pred_time]))
                    val_y_mort = np.append(val_y_mort, self.remove_padding(mort_labels[:, mort_pred_time],
                                                                           mask.type(self.bool_type)[:, mort_pred_time]))

            print('Validation Metrics:')
            mean_val_loss = sum(val_loss) / len(val_loss)
            if self.config.task in ('LoS', 'multitask'):
                los_metrics_list = print_metrics_regression(val_y_los, val_y_hat_los, elog=self.elog) # order: mad, mse, mape, msle, r2, kappa
                for metric_name, metric in zip(['mad', 'mse', 'mape', 'msle', 'r2', 'kappa'], los_metrics_list):
                    self.add_result(value=metric, name='val_' + metric_name, counter=epoch)
            if self.config.task in ('mortality', 'multitask'):
                mort_metrics_list = print_metrics_mortality(val_y_mort, val_y_hat_mort, elog=self.elog)
                for metric_name, metric in zip(['acc', 'prec0', 'prec1', 'rec0', 'rec1', 'auroc', 'auprc', 'f1macro'], mort_metrics_list):
                    self.add_result(value=metric, name='val_' + metric_name, counter=epoch)
            self.elog.print('Epoch: {} | Validation Loss: {:3.4f}'.format(epoch, mean_val_loss))

        elif self.config.mode == 'test' and epoch == self.n_epochs - 1:
            self.test()

        if epoch == self.n_epochs - 1 and self.config.save_results_csv:
            if self.config.task in ('LoS', 'multitask'):
                self.elog.save_to_csv(np.vstack((val_y_hat_los, val_y_los)).transpose(),
                                      'val_predictions_los/epoch{}.csv'.format(epoch),
                                      header='los_predictions, label')
            if self.config.task in ('mortality', 'multitask'):
                self.elog.save_to_csv(np.vstack((val_y_hat_mort, val_y_mort)).transpose(),
                                      'val_predictions_mort/epoch{}.csv'.format(epoch),
                                  header='mort_predictions, label')

        return

    def test(self, mort_pred_time=24):

        self.model.eval()
        test_batches = self.test_datareader.batch_gen(batch_size=self.config.batch_size_test)
        test_loss = []
        test_y_hat_los = np.array([])
        test_y_los = np.array([])
        test_y_hat_mort = np.array([])
        test_y_mort = np.array([])

        for batch in test_batches:

            # unpack batch
            if self.config.dataset == 'MIMIC':
                padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
                diagnoses = None
            else:
                padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch

            y_hat_los, y_hat_mort = self.model(padded, diagnoses, flat)
            loss = self.model.loss(y_hat_los, y_hat_mort, los_labels, mort_labels, mask, seq_lengths, self.device,
                                   self.config.sum_losses, self.config.loss)
            test_loss.append(loss.item())  # can't add the model.loss directly because it causes a memory leak

            if self.config.task in ('LoS', 'multitask'):
                test_y_hat_los = np.append(test_y_hat_los,
                                          self.remove_padding(y_hat_los, mask.type(self.bool_type)))
                test_y_los = np.append(test_y_los, self.remove_padding(los_labels, mask.type(self.bool_type)))
            if self.config.task in ('mortality', 'multitask') and mort_labels.shape[1] >= mort_pred_time:
                test_y_hat_mort = np.append(test_y_hat_mort,
                                           self.remove_padding(y_hat_mort[:, mort_pred_time],
                                                               mask.type(self.bool_type)[:, mort_pred_time]))
                test_y_mort = np.append(test_y_mort, self.remove_padding(mort_labels[:, mort_pred_time],
                                                                         mask.type(self.bool_type)[:, mort_pred_time]))

        print('Test Metrics:')
        mean_test_loss = sum(test_loss) / len(test_loss)

        if self.config.task in ('LoS', 'multitask'):
            los_metrics_list = print_metrics_regression(test_y_los, test_y_hat_los, elog=self.elog)  # order: mad, mse, mape, msle, r2, kappa
            for metric_name, metric in zip(['mad', 'mse', 'mape', 'msle', 'r2', 'kappa'], los_metrics_list):
                self.add_result(value=metric, name='test_' + metric_name)
        if self.config.task in ('mortality', 'multitask'):
            mort_metrics_list = print_metrics_mortality(test_y_mort, test_y_hat_mort, elog=self.elog)
            for metric_name, metric in zip(['acc', 'prec0', 'prec1', 'rec0', 'rec1', 'auroc', 'auprc', 'f1macro'],
                                           mort_metrics_list):
                self.add_result(value=metric, name='test_' + metric_name)

        if self.config.save_results_csv:
            if self.config.task in ('LoS', 'multitask'):
                self.elog.save_to_csv(np.vstack((test_y_hat_los, test_y_los)).transpose(), 'val_predictions_los.csv', header='los_predictions, label')
            if self.config.task in ('mortality', 'multitask'):
                self.elog.save_to_csv(np.vstack((test_y_hat_mort, test_y_mort)).transpose(), 'val_predictions_mort.csv', header='mort_predictions, label')
        self.elog.print('Test Loss: {:3.4f}'.format(mean_test_loss))

        # write to file
        if self.config.task == 'LoS':
            with open(self.config.base_dir + '/results.csv', 'a') as f:
                values = self.elog.plot_logger.values
                mad = values['test_mad']['test_mad'][-1][0]
                mse = values['test_mse']['test_mse'][-1][0]
                mape = values['test_mape']['test_mape'][-1][0]
                msle = values['test_msle']['test_msle'][-1][0]
                r2 = values['test_r2']['test_r2'][-1][0]
                kappa = values['test_kappa']['test_kappa'][-1][0]
                f.write('\n{},{},{},{},{},{}'.format(mad, mse, mape, msle, r2, kappa))
        elif self.config.task == 'mortality':
            with open(self.config.base_dir + '/results.csv', 'a') as f:
                values = self.elog.plot_logger.values
                acc = values['test_acc']['test_acc'][-1][0]
                prec0 = values['test_prec0']['test_prec0'][-1][0]
                prec1 = values['test_prec1']['test_prec1'][-1][0]
                rec0 = values['test_rec0']['test_rec0'][-1][0]
                rec1 = values['test_rec1']['test_rec1'][-1][0]
                auroc = values['test_auroc']['test_auroc'][-1][0]
                auprc = values['test_auprc']['test_auprc'][-1][0]
                f1macro = values['test_f1macro']['test_f1macro'][-1][0]
                f.write('\n{},{},{},{},{},{},{},{}'.format(acc, prec0, prec1, rec0, rec1, auroc, auprc, f1macro))
        elif self.config.task == 'multitask':
            with open(self.config.base_dir + '/results.csv', 'a') as f:
                values = self.elog.plot_logger.values
                mad = values['test_mad']['test_mad'][-1][0]
                mse = values['test_mse']['test_mse'][-1][0]
                mape = values['test_mape']['test_mape'][-1][0]
                msle = values['test_msle']['test_msle'][-1][0]
                r2 = values['test_r2']['test_r2'][-1][0]
                kappa = values['test_kappa']['test_kappa'][-1][0]
                acc = values['test_acc']['test_acc'][-1][0]
                prec0 = values['test_prec0']['test_prec0'][-1][0]
                prec1 = values['test_prec1']['test_prec1'][-1][0]
                rec0 = values['test_rec0']['test_rec0'][-1][0]
                rec1 = values['test_rec1']['test_rec1'][-1][0]
                auroc = values['test_auroc']['test_auroc'][-1][0]
                auprc = values['test_auprc']['test_auprc'][-1][0]
                f1macro = values['test_f1macro']['test_f1macro'][-1][0]
                f.write('\n{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(mad, mse, mape, msle, r2, kappa, acc, prec0, prec1, rec0, rec1, auroc, auprc, f1macro))
        return

    def resume(self, epoch):
        return
