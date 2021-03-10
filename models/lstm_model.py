import torch.nn as nn
import torch
from models.tpc_model import MSLELoss, MSELoss, MyBatchNorm1d, EmptyModule
from torch import exp, cat


class BaseLSTM(nn.Module):
    def __init__(self, config, F=None, D=None, no_flat_features=None):

        # The timeseries data will be of dimensions B * (2F + 2) * T where:
        #   B is the batch size
        #   F is the number of features for convolution (N.B. we start with 2F because there are corresponding mask features)
        #   T is the number of timepoints
        #   The other 2 features represent the sequence number and the hour in the day

        # The diagnoses data will be of dimensions B * D where:
        #   D is the number of diagnoses
        # The flat data will be of dimensions B * no_flat_features

        super(BaseLSTM, self).__init__()
        self.task = config.task
        self.hidden_size = config.hidden_size
        self.bidirectional = config.bidirectional
        self.channelwise = config.channelwise
        self.n_layers = config.n_layers
        self.lstm_dropout_rate = config.lstm_dropout_rate
        self.main_dropout_rate = config.main_dropout_rate
        self.diagnosis_size = config.diagnosis_size
        self.batchnorm = config.batchnorm
        self.last_linear_size = config.last_linear_size
        self.n_layers = config.n_layers
        self.F = F
        self.D = D
        self.no_flat_features = no_flat_features
        self.no_exp = config.no_exp
        self.alpha = config.alpha
        self.momentum = 0.01 if self.batchnorm == 'low_momentum' else 0.1
        self.no_diag = config.no_diag

        self.n_units = self.hidden_size // 2 if self.bidirectional else self.hidden_size
        self.n_dir = 2 if self.bidirectional else 1

        # use the same initialisation as in keras
        for m in self.modules():
            self.init_weights(m)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hardtanh = nn.Hardtanh(min_val=1 / 48, max_val=100)  # keep the end predictions between half an hour and 100 days
        self.lstm_dropout = nn.Dropout(p=self.lstm_dropout_rate)
        self.main_dropout = nn.Dropout(p=self.main_dropout_rate)
        self.msle_loss = MSLELoss()
        self.mse_loss = MSELoss()
        self.bce_loss = nn.BCELoss()

        self.empty_module = EmptyModule()
        self.remove_none = lambda x: tuple(xi for xi in x if xi is not None)

        if self.channelwise is False:
            # note if it's bidirectional, then we can't assume there's no influence from future timepoints on past ones
            self.lstm = nn.LSTM(input_size=(2*self.F + 2), hidden_size=self.n_units, num_layers=self.n_layers,
                                bidirectional=self.bidirectional, dropout=self.lstm_dropout_rate)
        elif self.channelwise:
            self.channelwise_lstm_list = nn.ModuleList([nn.LSTM(input_size=2, hidden_size=self.n_units,
                                                                num_layers=self.n_layers, bidirectional=self.bidirectional,
                                                                dropout=self.lstm_dropout_rate) for i in range(self.F)])

        # input shape: B * D
        # output shape: B * diagnosis_size
        self.diagnosis_encoder = nn.Linear(in_features=self.D, out_features=self.diagnosis_size)

        # input shape: B * diagnosis_size
        if self.batchnorm in ['mybatchnorm', 'low_momentum']:
            self.bn_diagnosis_encoder = MyBatchNorm1d(num_features=self.diagnosis_size, momentum=self.momentum)
        elif self.batchnorm == 'default':
            self.bn_diagnosis_encoder = nn.BatchNorm1d(num_features=self.diagnosis_size)
        else:
            self.bn_diagnosis_encoder = self.empty_module

        # input shape: (B * T) * (n_units + diagnosis_size + no_flat_features)
        # output shape: (B * T) * last_linear_size
        channel_wise = self.F if self.channelwise else 1
        input_size = self.n_units * channel_wise + self.diagnosis_size + self.no_flat_features
        if self.no_diag:
            input_size = input_size - self.diagnosis_size
        self.point_los = nn.Linear(in_features=input_size, out_features=self.last_linear_size)
        self.point_mort = nn.Linear(in_features=input_size, out_features=self.last_linear_size)

        # input shape: (B * T) * last_linear_size
        if self.batchnorm in ['mybatchnorm', 'pointonly', 'low_momentum']:
            self.bn_point_last_los = MyBatchNorm1d(num_features=self.last_linear_size, momentum=self.momentum)
            self.bn_point_last_mort = MyBatchNorm1d(num_features=self.last_linear_size, momentum=self.momentum)
        elif self.batchnorm == 'default':
            self.bn_point_last_los = nn.BatchNorm1d(num_features=self.last_linear_size)
            self.bn_point_last_mort = nn.BatchNorm1d(num_features=self.last_linear_size)
        else:
            self.bn_point_last_los = self.empty_module
            self.bn_point_last_mort = self.empty_module

        # input shape: (B * T) * last_linear_size
        # output shape: (B * T) * 1
        self.point_final_los = nn.Linear(in_features=self.last_linear_size, out_features=1)
        self.point_final_mort = nn.Linear(in_features=self.last_linear_size, out_features=1)

        return

    def init_weights(self, m):
        if isinstance(m, nn.LSTM):
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            for names in m._all_weights:
                for name in filter(lambda n: 'bias' in n, names):
                    bias = getattr(m, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)
        return

    def init_hidden(self, B, device):
        h0 = torch.zeros(self.n_layers * self.n_dir, B, self.n_units).to(device)
        c0 = torch.zeros(self.n_layers * self.n_dir, B, self.n_units).to(device)
        return (h0, c0)

    def forward(self, X, diagnoses, flat, time_before_pred=5):

        # flat is B * no_flat_features
        # diagnoses is B * D
        # X is B * (2F + 2) * T
        # X_mask is B * T
        # (the batch is padded to the longest sequence)

        B, _, T = X.shape

        if self.channelwise is False:
            # the lstm expects (seq_len, batch, input_size)
            # N.B. the default hidden state is zeros so we don't need to specify it
            lstm_output, hidden = self.lstm(X.permute(2, 0, 1))  # T * B * hidden_size

        elif self.channelwise is True:
            # take time and hour fields as they are not useful when processed on their own (they go up linearly. They were also taken out for temporal convolution so the comparison is fair)
            X_separated = torch.split(X[:, 1:-1, :], self.F, dim=1)  # tuple ((B * F * T), (B * F * T))
            X_rearranged = torch.stack(X_separated, dim=2)  # B * F * 2 * T
            lstm_output = None
            for i in range(self.F):
                X_lstm, hidden = self.channelwise_lstm_list[i](X_rearranged[:, i, :, :].permute(2, 0, 1))
                lstm_output = cat(self.remove_none((lstm_output, X_lstm)), dim=2)

        X_final = self.relu(self.lstm_dropout(lstm_output.permute(1, 2, 0)))

        # note that we cut off at time_before_pred hours here because the model is only valid from time_before_pred hours onwards
        if self.no_diag:
            combined_features = cat((flat.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * no_flat_features
                                     X_final[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)), dim=1)
        else:
            diagnoses_enc = self.relu(self.main_dropout(self.bn_diagnosis_encoder(self.diagnosis_encoder(diagnoses))))  # B * diagnosis_size
            combined_features = cat((flat.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * no_flat_features
                                     diagnoses_enc.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * diagnosis_size
                                     X_final[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)), dim=1)

        last_point_los = self.relu(self.main_dropout(self.bn_point_last_los(self.point_los(combined_features))))
        last_point_mort = self.relu(self.main_dropout(self.bn_point_last_mort(self.point_mort(combined_features))))

        if self.no_exp:
            los_predictions = self.hardtanh(self.point_final_los(last_point_los).view(B, T - time_before_pred))  # B * (T - time_before_pred)
        else:
            los_predictions = self.hardtanh(exp(self.point_final_los(last_point_los).view(B, T - time_before_pred)))  # B * (T - time_before_pred)
        mort_predictions = self.sigmoid(self.point_final_mort(last_point_mort).view(B, T - time_before_pred))  # B * (T - time_before_pred)

        return los_predictions, mort_predictions

    def loss(self, y_hat_los, y_hat_mort, y_los, y_mort, mask, seq_lengths, device, sum_losses, loss_type):
        # mort loss
        if self.task == 'mortality':
            loss = self.bce_loss(y_hat_mort, y_mort) * self.alpha
        # los loss
        else:
            bool_type = torch.cuda.BoolTensor if device == torch.device('cuda') else torch.BoolTensor
            if loss_type == 'msle':
                los_loss = self.msle_loss(y_hat_los, y_los, mask.type(bool_type), seq_lengths, sum_losses)
            elif loss_type == 'mse':
                los_loss = self.mse_loss(y_hat_los, y_los, mask.type(bool_type), seq_lengths, sum_losses)
            if self.task == 'LoS':
                loss = los_loss
            # multitask loss
            if self.task == 'multitask':
                loss = los_loss + self.bce_loss(y_hat_mort, y_mort) * self.alpha
        return loss