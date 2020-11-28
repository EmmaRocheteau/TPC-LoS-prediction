import torch
import pandas as pd
from itertools import groupby, islice
import numpy as np

# bit hacky but passes checks and I don't have time to implement a neater solution
lab_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 15, 16, 18, 21, 22, 23, 24, 29, 32, 33, 34, 39, 40, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 63, 67, 68, 69, 70, 71, 72, 75, 83, 84, 86]
labs_to_keep = [0] + [(i + 1) for i in lab_indices] + [(i + 88) for i in lab_indices] + [-1]
no_lab_indices = list(range(87))
no_lab_indices = [x for x in no_lab_indices if x not in lab_indices]
no_labs_to_keep = [0] + [(i + 1) for i in no_lab_indices] + [(i + 88) for i in no_lab_indices] + [-1]


class eICUReader(object):

    def __init__(self, data_path, device=None, labs_only=False, no_labs=False):
        self._diagnoses_path = data_path + '/diagnoses.csv'
        self._labels_path = data_path + '/labels.csv'
        self._flat_path = data_path + '/flat.csv'
        self._timeseries_path = data_path + '/timeseries.csv'
        self._device = device
        self.labs_only = labs_only
        self.no_labs = no_labs
        self._dtype = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor

        self.labels = pd.read_csv(self._labels_path, index_col='patient')
        self.flat = pd.read_csv(self._flat_path, index_col='patient')
        self.diagnoses = pd.read_csv(self._diagnoses_path, index_col='patient')

        # we minus 2 to calculate F because hour and time are not features for convolution
        self.F = (pd.read_csv(self._timeseries_path, index_col='patient', nrows=1).shape[1] - 2)//2
        self.D = self.diagnoses.shape[1]
        self.no_flat_features = self.flat.shape[1]

        self.patients = list(self.labels.index)
        self.no_patients = len(self.patients)

    def line_split(self, line):
        return [float(x) for x in line.split(',')]

    def pad_sequences(self, ts_batch):
        seq_lengths = [len(x) for x in ts_batch]
        max_len = max(seq_lengths)
        padded = [patient + [[0] * (self.F * 2 + 2)] * (max_len - len(patient)) for patient in ts_batch]
        if self.labs_only:
            padded = np.array(padded)
            padded = padded[:, :, labs_to_keep]
        if self.no_labs:
            padded = np.array(padded)
            padded = padded[:, :, no_labs_to_keep]
        padded = torch.tensor(padded, device=self._device).type(self._dtype).permute(0, 2, 1)  # B * (2F + 2) * T
        padded[:, 0, :] /= 24  # scale the time into days instead of hours
        mask = torch.zeros(padded[:, 0, :].shape, device=self._device).type(self._dtype)
        for p, l in enumerate(seq_lengths):
            mask[p, :l] = 1
        return padded, mask, torch.tensor(seq_lengths).type(self._dtype)

    def get_los_labels(self, labels, times, mask):
        times = labels.unsqueeze(1).repeat(1, times.shape[1]) - times
        # clamp any labels that are less than 30 mins otherwise it becomes too small when the log is taken
        # make sure where there is no data the label is 0
        return (times.clamp(min=1/48) * mask)

    def get_mort_labels(self, labels, length):
        repeated_labels = labels.unsqueeze(1).repeat(1, length)
        return repeated_labels

    def batch_gen(self, batch_size=8, time_before_pred=5):

        # note that once the generator is finished, the file will be closed automatically
        with open(self._timeseries_path, 'r') as timeseries_file:
            # the first line is the feature names; we have to skip over this
            self.timeseries_header = next(timeseries_file).strip().split(',')
            # this produces a generator that returns a list of batch_size patient identifiers
            patient_batches = (self.patients[pos:pos + batch_size] for pos in range(0, len(self.patients), batch_size))
            # create a generator to capture a single patient timeseries
            ts_patient = groupby(map(self.line_split, timeseries_file), key=lambda line: line[0])
            # we loop through these batches, tracking the index because we need it to index the pandas dataframes
            for i, batch in enumerate(patient_batches):
                ts_batch = [[line[1:] for line in ts] for _, ts in islice(ts_patient, batch_size)]
                padded, mask, seq_lengths = self.pad_sequences(ts_batch)
                los_labels = self.get_los_labels(torch.tensor(self.labels.iloc[i*batch_size:(i+1)*batch_size,7].values, device=self._device).type(self._dtype), padded[:,0,:], mask)
                mort_labels = self.get_mort_labels(torch.tensor(self.labels.iloc[i*batch_size:(i+1)*batch_size,5].values, device=self._device).type(self._dtype), length=mask.shape[1])

                # we must avoid taking data before time_before_pred hours to avoid diagnoses and apache variable from the future
                yield (padded,  # B * (2F + 2) * T
                       mask[:, time_before_pred:],  # B * (T - time_before_pred)
                       torch.tensor(self.diagnoses.iloc[i*batch_size:(i+1)*batch_size].values, device=self._device).type(self._dtype),  # B * D
                       torch.tensor(self.flat.iloc[i*batch_size:(i+1)*batch_size].values.astype(float), device=self._device).type(self._dtype),  # B * no_flat_features
                       los_labels[:, time_before_pred:],
                       mort_labels[:, time_before_pred:],
                       seq_lengths - time_before_pred)

if __name__=='__main__':
    eICU_reader = eICUReader('/Users/emmarocheteau/PycharmProjects/TPC-LoS-prediction/eICU_data/train')
    print(next(eICU_reader.batch_gen()))