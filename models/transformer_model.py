import math
import torch
import torch.nn as nn
from models.tpc_model import MSLELoss, MSELoss, MyBatchNorm1d, EmptyModule
from torch import exp, cat


# PositionalEncoding adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html. I made the following
# changes:
    # Took out the dropout
    # Changed the dimensions/shape of pe
# I am using the positional encodings suggested by Vaswani et al. as the Attend and Diagnose authors do not specify in
# detail how they do their positional encodings.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=14*24):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)  # changed from max_len * d_model to 1 * d_model * max_len
        self.register_buffer('pe', pe)

    def forward(self, X):
        # X is B * d_model * T
        # self.pe[:, :, :X.size(2)] is 1 * d_model * T but is broadcast to B when added
        X = X + self.pe[:, :, :X.size(2)]  # B * d_model * T
        return X  # B * d_model * T


class TransformerEncoder(nn.Module):
    def __init__(self, input_size=None, d_model=None, num_layers=None, num_heads=None, feedforward_size=None, dropout=None,
                 pe=None, device=None):
        super(TransformerEncoder, self).__init__()

        self.device = device
        self.d_model = d_model
        self.pe = pe  # boolean variable indicating whether or not the positional encoding should be applied
        self.input_embedding = nn.Conv1d(in_channels=input_size, out_channels=d_model, kernel_size=1)  # B * C * T
        self.pos_encoder = PositionalEncoding(d_model)
        self.trans_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                                              dim_feedforward=feedforward_size, dropout=dropout,
                                                              activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.trans_encoder_layer, num_layers=num_layers)

    def _causal_mask(self, size=None):
        mask = (torch.triu(torch.ones(size, size).to(self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask  # T * T

    def forward(self, X, T):
        # X is B * (2F + 2) * T

        # multiplication by root(d_model) as described in Vaswani et al. 2017 section 3.4
        X = self.input_embedding(X) * math.sqrt(self.d_model)  # B * d_model * T
        if self.pe:  # apply the positional encoding
            X = self.pos_encoder(X)  # B * d_model * T
        X = self.transformer_encoder(src=X.permute(2, 0, 1), mask=self._causal_mask(size=T))  # T * B * d_model
        return X.permute(1, 2, 0)  # B * d_model * T


class Transformer(nn.Module):
    def __init__(self, config, F=None, D=None, no_flat_features=None, device=None):

        # The timeseries data will be of dimensions B * (2F + 2) * T where:
        #   B is the batch size
        #   F is the number of features for convolution (N.B. we start with 2F because there are corresponding mask features)
        #   T is the number of timepoints
        #   The other 2 features represent the sequence number and the hour in the day

        # The diagnoses data will be of dimensions B * D where:
        #   D is the number of diagnoses
        # The flat data will be of dimensions B * no_flat_features

        super(Transformer, self).__init__()
        self.task = config.task
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.feedforward_size = config.feedforward_size
        self.trans_dropout_rate = config.trans_dropout_rate
        self.positional_encoding = config.positional_encoding
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

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hardtanh = nn.Hardtanh(min_val=1 / 48, max_val=100)  # keep the end predictions between half an hour and 100 days
        self.trans_dropout = nn.Dropout(p=self.trans_dropout_rate)
        self.main_dropout = nn.Dropout(p=self.main_dropout_rate)
        self.msle_loss = MSLELoss()
        self.mse_loss = MSELoss()
        self.bce_loss = nn.BCELoss()

        self.empty_module = EmptyModule()
        self.remove_none = lambda x: tuple(xi for xi in x if xi is not None)

        self.transformer = TransformerEncoder(input_size=(2*self.F + 2), d_model=self.d_model, num_layers=self.n_layers,
                                              num_heads=self.n_heads, feedforward_size=self.feedforward_size,
                                              dropout=self.trans_dropout_rate, pe=self.positional_encoding,
                                              device=device)

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

        # input shape: (B * T) * (d_model + diagnosis_size + no_flat_features)
        # output shape: (B * T) * last_linear_size
        input_size = self.d_model + self.diagnosis_size + self.no_flat_features
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

    def forward(self, X, diagnoses, flat, time_before_pred=5):

        # flat is B * no_flat_features
        # diagnoses is B * D
        # X is B * (2F + 2) * T
        # X_mask is B * T
        # (the batch is padded to the longest sequence)

        B, _, T = X.shape  # B * (2F + 2) * T

        trans_output = self.transformer(X, T)  # B * d_model * T

        X_final = self.relu(self.trans_dropout(trans_output))  # B * d_model * T

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