import torch
import torch.nn as nn
from torch import cat, exp
import torch.nn.functional as F
from torch.nn.functional import pad
from torch.nn.modules.batchnorm import _BatchNorm


###============== The main defining function of the TPC model is temp_pointwise() on line 403 ==============###


# Mean Squared Logarithmic Error (MSLE) loss
class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.squared_error = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y, mask, seq_length, sum_losses=False):
        # the log(predictions) corresponding to no data should be set to 0
        log_y_hat = y_hat.log().where(mask, torch.zeros_like(y))
        # the we set the log(labels) that correspond to no data to be 0 as well
        log_y = y.log().where(mask, torch.zeros_like(y))
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(log_y_hat, log_y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1)
        return loss.mean()


# Mean Squared Error (MSE) loss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.squared_error = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y, mask, seq_length, sum_losses=False):
        # the predictions corresponding to no data should be set to 0
        y_hat = y_hat.where(mask, torch.zeros_like(y))
        # the we set the labels that correspond to no data to be 0 as well
        y = y.where(mask, torch.zeros_like(y))
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(y_hat, y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1)
        return loss.mean()


class MyBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MyBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        # hack to work around model.eval() issue
        if not self.training:
            self.eval_momentum = 0  # set the momentum to zero when the model is validating

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum if self.training else self.eval_momentum

        if self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum if self.training else self.eval_momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            training=True, momentum=exponential_average_factor, eps=self.eps)  # set training to True so it calculates the norm of the batch


class MyBatchNorm1d(MyBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class EmptyModule(nn.Module):
    def forward(self, X):
        return X


class TempPointConv(nn.Module):
    def __init__(self, config, F=None, D=None, no_flat_features=None):

        # The timeseries data will be of dimensions B * (2F + 2) * T where:
        #   B is the batch size
        #   F is the number of features for convolution (N.B. we start with 2F because there are corresponding mask features)
        #   T is the number of timepoints
        #   The other 2 features represent the sequence number and the hour in the day

        # The diagnoses data will be of dimensions B * D where:
        #   D is the number of diagnoses
        # The flat data will be of dimensions B * no_flat_features

        super(TempPointConv, self).__init__()
        self.task = config.task
        self.n_layers = config.n_layers
        self.model_type = config.model_type
        self.share_weights = config.share_weights
        self.diagnosis_size = config.diagnosis_size
        self.main_dropout_rate = config.main_dropout_rate
        self.temp_dropout_rate = config.temp_dropout_rate
        self.kernel_size = config.kernel_size
        self.temp_kernels = config.temp_kernels
        self.point_sizes = config.point_sizes
        self.batchnorm = config.batchnorm
        self.last_linear_size = config.last_linear_size
        self.F = F
        self.D = D
        self.no_flat_features = no_flat_features
        self.no_diag = config.no_diag
        self.no_mask = config.no_mask
        self.no_exp = config.no_exp
        self.no_skip_connections = config.no_skip_connections
        self.alpha = config.alpha
        self.momentum = 0.01 if self.batchnorm == 'low_momentum' else 0.1

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hardtanh = nn.Hardtanh(min_val=1/48, max_val=100)  # keep the end predictions between half an hour and 100 days
        self.msle_loss = MSLELoss()
        self.mse_loss = MSELoss()
        self.bce_loss = nn.BCELoss()

        self.main_dropout = nn.Dropout(p=self.main_dropout_rate)
        self.temp_dropout = nn.Dropout(p=self.temp_dropout_rate)

        self.remove_none = lambda x: tuple(xi for xi in x if xi is not None)  # removes None items from a tuple
        self.empty_module = EmptyModule()

        if self.batchnorm in ['mybatchnorm', 'pointonly', 'temponly', 'low_momentum']:
            self.batchnormclass = MyBatchNorm1d
        elif self.batchnorm == 'default':
            self.batchnormclass = nn.BatchNorm1d

        # input shape: B * D
        # output shape: B * diagnosis_size
        self.diagnosis_encoder = nn.Linear(in_features=self.D, out_features=self.diagnosis_size)

        if self.batchnorm in ['mybatchnorm', 'pointonly', 'low_momentum', 'default']:
            self.bn_diagnosis_encoder = self.batchnormclass(num_features=self.diagnosis_size, momentum=self.momentum)  # input shape: B * diagnosis_size
            self.bn_point_last_los = self.batchnormclass(num_features=self.last_linear_size, momentum=self.momentum)  # input shape: (B * T) * last_linear_size
            self.bn_point_last_mort = self.batchnormclass(num_features=self.last_linear_size, momentum=self.momentum)  # input shape: (B * T) * last_linear_size
        else:
            self.bn_diagnosis_encoder = self.empty_module
            self.bn_point_last_los = self.empty_module
            self.bn_point_last_mort = self.empty_module

        # input shape: (B * T) * last_linear_size
        # output shape: (B * T) * 1
        self.point_final_los = nn.Linear(in_features=self.last_linear_size, out_features=1)
        self.point_final_mort = nn.Linear(in_features=self.last_linear_size, out_features=1)

        if self.model_type == 'tpc':
            self.init_tpc()
        elif self.model_type == 'temp_only':
            self.init_temp()
        elif self.model_type == 'pointwise_only':
            self.init_pointwise()
        else:
            raise NotImplementedError('Specified model type not supported; supported types include tpc, temp_only and pointwise_only')


    def init_tpc(self):

        # non-module layer attributes
        self.layers = []
        for i in range(self.n_layers):
            dilation = i * (self.kernel_size - 1) if i > 0 else 1  # dilation = 1 for the first layer, after that it captures all the information gathered by previous layers
            temp_k = self.temp_kernels[i]
            point_size = self.point_sizes[i]
            self.update_layer_info(layer=i, temp_k=temp_k, point_size=point_size, dilation=dilation, stride=1)

        # module layer attributes
        self.create_temp_pointwise_layers()

        # input shape: (B * T) * ((F + Zt) * (1 + Y) + diagnosis_size + no_flat_features)
        # output shape: (B * T) * last_linear_size
        input_size = (self.F + self.Zt) * (1 + self.Y) + self.diagnosis_size + self.no_flat_features
        if self.no_diag:
            input_size = input_size - self.diagnosis_size
        if self.no_skip_connections:
            input_size = self.F * self.Y + self.Z + self.diagnosis_size + self.no_flat_features
        self.point_last_los = nn.Linear(in_features=input_size, out_features=self.last_linear_size)
        self.point_last_mort = nn.Linear(in_features=input_size, out_features=self.last_linear_size)

        return


    def init_temp(self):

        # non-module layer attributes
        self.layers = []
        for i in range(self.n_layers):
            dilation = i * (self.kernel_size - 1) if i > 0 else 1  # dilation = 1 for the first layer, after that it captures all the information gathered by previous layers
            temp_k = self.temp_kernels[i]
            self.update_layer_info(layer=i, temp_k=temp_k, dilation=dilation, stride=1)

        # module layer attributes
        self.create_temp_only_layers()

        # input shape: (B * T) * (F * (1 + Y) + diagnosis_size + no_flat_features)
        # output shape: (B * T) * last_linear_size
        input_size = self.F * (1 + self.Y) + self.diagnosis_size + self.no_flat_features
        self.point_last_los = nn.Linear(in_features=input_size, out_features=self.last_linear_size)
        self.point_last_mort = nn.Linear(in_features=input_size, out_features=self.last_linear_size)
        return


    def init_pointwise(self):

        # non-module layer attributes
        self.layers = []
        for i in range(self.n_layers):
            point_size = self.point_sizes[i]
            self.update_layer_info(layer=i, point_size=point_size)

        # module layer attributes
        self.create_pointwise_only_layers()

        # input shape: (B * T) * (Zt + 2F + 2 + no_flat_features + diagnosis_size)
        # output shape: (B * T) * last_linear_size
        if self.no_mask:
            input_size = self.Zt + self.F + 2 + self.no_flat_features + self.diagnosis_size
        else:
            input_size = self.Zt + 2 * self.F + 2 + self.no_flat_features + self.diagnosis_size
        self.point_last_los = nn.Linear(in_features=input_size, out_features=self.last_linear_size)
        self.point_last_mort = nn.Linear(in_features=input_size, out_features=self.last_linear_size)

        return


    def update_layer_info(self, layer=None, temp_k=None, point_size=None, dilation=None, stride=None):

        self.layers.append({})
        if point_size is not None:
            self.layers[layer]['point_size'] = point_size
        if temp_k is not None:
            padding = [(self.kernel_size - 1) * dilation, 0]  # [padding_left, padding_right]
            self.layers[layer]['temp_kernels'] = temp_k
            self.layers[layer]['dilation'] = dilation
            self.layers[layer]['padding'] = padding
            self.layers[layer]['stride'] = stride

        return


    def create_temp_pointwise_layers(self):

        ### Notation used for tracking the tensor shapes ###

        # Z is the number of extra features added by the previous pointwise layer (could be 0 if this is the first layer)
        # Zt is the cumulative number of extra features that have been added by all previous pointwise layers
        # Zt-1 = Zt - Z (cumulative number of extra features minus the most recent pointwise layer)
        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)

        self.layer_modules = nn.ModuleDict()

        self.Y = 0
        self.Z = 0
        self.Zt = 0

        for i in range(self.n_layers):

            temp_in_channels = (self.F + self.Zt) * (1 + self.Y) if i > 0 else 2 * self.F  # (F + Zt) * (Y + 1)
            temp_out_channels = (self.F + self.Zt) * self.layers[i]['temp_kernels']  # (F + Zt) * temp_kernels
            linear_input_dim = (self.F + self.Zt - self.Z) * self.Y + self.Z + 2 * self.F + 2 + self.no_flat_features  # (F + Zt-1) * Y + Z + 2F + 2 + no_flat_features
            linear_output_dim = self.layers[i]['point_size']  # point_size
            # correct if no_mask
            if self.no_mask:
                if i == 0:
                    temp_in_channels = self.F
                linear_input_dim = (self.F + self.Zt - self.Z) * self.Y + self.Z + self.F + 2 + self.no_flat_features  # (F + Zt-1) * Y + Z + F + 2 + no_flat_features

            temp = nn.Conv1d(in_channels=temp_in_channels,  # (F + Zt) * (Y + 1)
                             out_channels=temp_out_channels,  # (F + Zt) * Y
                             kernel_size=self.kernel_size,
                             stride=self.layers[i]['stride'],
                             dilation=self.layers[i]['dilation'],
                             groups=self.F + self.Zt)

            point = nn.Linear(in_features=linear_input_dim, out_features=linear_output_dim)

            # correct if no_skip_connections
            if self.no_skip_connections:
                temp_in_channels = self.F * self.Y if i > 0 else 2 * self.F  # F * Y
                temp_out_channels = self.F * self.layers[i]['temp_kernels']  # F * temp_kernels
                #linear_input_dim = self.F * self.Y + self.Z if i > 0 else 2 * self.F + 2 + self.no_flat_features  # (F * Y) + Z
                linear_input_dim = self.Z if i > 0 else 2 * self.F + 2 + self.no_flat_features  # Z
                temp = nn.Conv1d(in_channels=temp_in_channels,
                                 out_channels=temp_out_channels,
                                 kernel_size=self.kernel_size,
                                 stride=self.layers[i]['stride'],
                                 dilation=self.layers[i]['dilation'],
                                 groups=self.F)

                point = nn.Linear(in_features=linear_input_dim, out_features=linear_output_dim)

            if self.batchnorm in ['default', 'mybatchnorm', 'low_momentum']:
                bn_temp = self.batchnormclass(num_features=temp_out_channels, momentum=self.momentum)
                bn_point = self.batchnormclass(num_features=linear_output_dim, momentum=self.momentum)
            elif self.batchnorm == 'temponly':
                bn_temp = self.batchnormclass(num_features=temp_out_channels)
                bn_point = self.empty_module
            elif self.batchnorm == 'pointonly':
                bn_temp = self.empty_module
                bn_point = self.batchnormclass(num_features=linear_output_dim)
            else:
                bn_temp = bn_point = self.empty_module  # linear module; does nothing

            self.layer_modules[str(i)] = nn.ModuleDict({
                'temp': temp,
                'bn_temp': bn_temp,
                'point': point,
                'bn_point': bn_point})

            self.Y = self.layers[i]['temp_kernels']
            self.Z = linear_output_dim
            self.Zt += self.Z

        return


    def create_temp_only_layers(self):

        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)
        self.layer_modules = nn.ModuleDict()
        self.Y = 0

        for i in range(self.n_layers):

            if self.share_weights:
                temp_in_channels = (1 + self.Y) if i > 0 else 2  # (Y + 1)
                temp_out_channels = self.layers[i]['temp_kernels']
                groups = 1
            else:
                temp_in_channels = self.F * (1 + self.Y) if i > 0 else 2 * self.F  # F * (Y + 1)
                temp_out_channels = self.F * self.layers[i]['temp_kernels']  # F * temp_kernels
                groups = self.F

            temp = nn.Conv1d(in_channels=temp_in_channels,
                             out_channels=temp_out_channels,
                             kernel_size=self.kernel_size,
                             stride=self.layers[i]['stride'],
                             dilation=self.layers[i]['dilation'],
                             groups=groups)

            if self.batchnorm in ['default', 'mybatchnorm', 'low_momentum', 'temponly']:
                bn_temp = self.batchnormclass(num_features=temp_out_channels, momentum=self.momentum)
            else:
                bn_temp = self.empty_module  # linear module; does nothing

            self.layer_modules[str(i)] = nn.ModuleDict({
                'temp': temp,
                'bn_temp': bn_temp})

            self.Y = self.layers[i]['temp_kernels']

        return


    def create_pointwise_only_layers(self):

        # Zt is the cumulative number of extra features that have been added by previous pointwise layers
        self.layer_modules = nn.ModuleDict()
        self.Zt = 0

        for i in range(self.n_layers):

            linear_input_dim = self.Zt + 2 * self.F + 2 + self.no_flat_features  # Zt + 2F + 2 + no_flat_features
            linear_output_dim = self.layers[i]['point_size']  # point_size

            if self.no_mask:
                linear_input_dim = self.Zt + self.F + 2 + self.no_flat_features  # Zt + 2F + 2 + no_flat_features

            point = nn.Linear(in_features=linear_input_dim, out_features=linear_output_dim)

            if self.batchnorm in ['default', 'mybatchnorm', 'low_momentum', 'pointonly']:
                bn_point = self.batchnormclass(num_features=linear_output_dim, momentum=self.momentum)
            else:
                bn_point = self.empty_module  # linear module; does nothing

            self.layer_modules[str(i)] = nn.ModuleDict({
                'point': point,
                'bn_point': bn_point})

            self.Zt += linear_output_dim

        return


    # This is really where the crux of TPC is defined. This function defines one TPC layer, as in Figure 3 in the paper:
    # https://arxiv.org/pdf/2007.09483.pdf
    def temp_pointwise(self, B=None, T=None, X=None, repeat_flat=None, X_orig=None, temp=None, bn_temp=None, point=None,
                       bn_point=None, temp_kernels=None, point_size=None, padding=None, prev_temp=None, prev_point=None,
                       point_skip=None):

        ### Notation used for tracking the tensor shapes ###

        # Z is the number of extra features added by the previous pointwise layer (could be 0 if this is the first layer)
        # Zt is the cumulative number of extra features that have been added by all previous pointwise layers
        # Zt-1 = Zt - Z (cumulative number of extra features minus the most recent pointwise layer)
        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)
        # X shape: B * ((F + Zt) * (Y + 1)) * T; N.B exception in the first layer where there are also mask features, in this case it is B * 2F * T
        # repeat_flat shape: (B * T) * no_flat_features
        # X_orig shape: (B * T) * (2F + 2)
        # prev_temp shape: (B * T) * ((F + Zt-1) * (Y + 1))
        # prev_point shape: (B * T) * Z

        Z = prev_point.shape[1] if prev_point is not None else 0

        X_padded = pad(X, padding, 'constant', 0)  # B * ((F + Zt) * (Y + 1)) * (T + padding)
        X_temp = self.temp_dropout(bn_temp(temp(X_padded)))  # B * ((F + Zt) * temp_kernels) * T

        X_concat = cat(self.remove_none((prev_temp,  # (B * T) * ((F + Zt-1) * Y)
                                         prev_point,  # (B * T) * Z
                                         X_orig,  # (B * T) * (2F + 2)
                                         repeat_flat)),  # (B * T) * no_flat_features
                       dim=1)  # (B * T) * (((F + Zt-1) * Y) + Z + 2F + 2 + no_flat_features)

        point_output = self.main_dropout(bn_point(point(X_concat)))  # (B * T) * point_size

        # point_skip input: B * (F + Zt-1) * T
        # prev_point: B * Z * T
        # point_skip output: B * (F + Zt) * T
        point_skip = cat((point_skip, prev_point.view(B, T, Z).permute(0, 2, 1)), dim=1) if prev_point is not None else point_skip

        temp_skip = cat((point_skip.unsqueeze(2),  # B * (F + Zt) * 1 * T
                         X_temp.view(B, point_skip.shape[1], temp_kernels, T)),  # B * (F + Zt) * temp_kernels * T
                        dim=2)  # B * (F + Zt) * (1 + temp_kernels) * T

        X_point_rep = point_output.view(B, T, point_size, 1).permute(0, 2, 3, 1).repeat(1, 1, (1 + temp_kernels), 1)  # B * point_size * (1 + temp_kernels) * T
        X_combined = self.relu(cat((temp_skip, X_point_rep), dim=1))  # B * (F + Zt) * (1 + temp_kernels) * T
        next_X = X_combined.view(B, (point_skip.shape[1] + point_size) * (1 + temp_kernels), T)  # B * ((F + Zt + point_size) * (1 + temp_kernels)) * T

        temp_output = X_temp.permute(0, 2, 1).contiguous().view(B * T, point_skip.shape[1] * temp_kernels)  # (B * T) * ((F + Zt) * temp_kernels)

        return (temp_output,  # (B * T) * ((F + Zt) * temp_kernels)
                point_output,  # (B * T) * point_size
                next_X,  # B * ((F + Zt) * (1 + temp_kernels)) * T
                point_skip)  # for keeping track of the point skip connections; B * (F + Zt) * T


    def temp(self, B=None, T=None, X=None, X_temp_orig=None, temp=None, bn_temp=None, temp_kernels=None, padding=None):

        ### Notation used for tracking the tensor shapes ###

        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)
        # X shape: B * (F * (Y + 1)) * T; N.B exception in the first layer where there are also mask features, in this case it is B * 2F * T
        # X_temp_orig shape: B * F * T

        X_padded = pad(X, padding, 'constant', 0)  # B * (F * (Y + 1)) * (T + padding)

        if self.share_weights:
            _, C, padded_length = X_padded.shape
            chans = int(C / self.F)
            X_temp = self.temp_dropout(bn_temp(temp(X_padded.view(B * self.F, chans, padded_length)))).view(B, (self.F * temp_kernels), T)  # B * (F * temp_kernels) * T
        else:
            X_temp = self.temp_dropout(bn_temp(temp(X_padded)))  # B * (F * temp_kernels) * T

        temp_skip = self.relu(cat((X_temp_orig.unsqueeze(2),  # B * F * 1 * T
                                   X_temp.view(B, self.F, temp_kernels, T)),  # B * F * temp_kernels * T
                                   dim=2))  # B * F * (1 + temp_kernels) * T

        next_X = temp_skip.view(B, (self.F * (1 + temp_kernels)), T)  # B * (F * (1 + temp_kernels)) * T

        return next_X  # B * (F * temp_kernels) * T


    def point(self, B=None, T=None, X=None, repeat_flat=None, X_orig=None, point=None, bn_point=None, point_skip=None):

        ### Notation used for tracking the tensor shapes ###

        # Z is the number of extra features added by the previous pointwise layer (could be 0 if this is the first layer)
        # Zt is the cumulative number of extra features that have been added by all previous pointwise layers
        # Zt-1 = Zt - Z (cumulative number of extra features minus the most recent pointwise layer)
        # X shape: B * (F + Zt) * T; N.B exception in the first layer where there are also mask features, in this case it is B * 2F * T
        # repeat_flat shape: (B * T) * no_flat_features
        # X_orig shape: (B * T) * (2F + 2)
        # prev_point shape: (B * T) * Z

        X_combined = cat((X, repeat_flat), dim=1)

        X_point = self.main_dropout(bn_point(point(X_combined)))  # (B * T) * point_size

        # point_skip input: B * Zt-1 * T
        # prev_point: B * Z * T
        # point_skip output: B * Zt * T
        point_skip = cat(self.remove_none((point_skip, X_point.view(B, T, -1).permute(0, 2, 1))), dim=1)

        # point_skip: B * Zt * T
        # X_orig: (B * T) * (2F + 2)
        # repeat_flat: (B * T) * no_flat_features
        # next_X: (B * T) * (Zt + 2F + 2 + no_flat_features)
        next_X = self.relu(cat((point_skip.permute(0, 2, 1).contiguous().view(B * T, -1), X_orig), dim=1))

        return (next_X,  # (B * T) * (Zt + 2F + 2 + no_flat_features)
                point_skip)  # for keeping track of the pointwise skip connections; B * Zt * T


    def temp_pointwise_no_skip(self, B=None, T=None, temp=None, bn_temp=None, point=None, bn_point=None, padding=None, prev_temp=None,
                               prev_point=None, temp_kernels=None, X_orig=None, repeat_flat=None):

        ### Temporal component ###

        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)
        # prev_temp shape: B * (F * Y) * T; N.B exception in the first layer where there are also mask features, in this case it is B * 2F * T

        X_padded = pad(prev_temp, padding, 'constant', 0)  # B * (F * Y) * (T + padding)
        temp_output = self.relu(self.temp_dropout(bn_temp(temp(X_padded))))  # B * (F * temp_kernels) * T

        ### Pointwise component ###

        # prev_point shape: (B * T) * ((F * Y) + Z)
        point_output = self.relu(self.main_dropout(bn_point(point(prev_point))))  # (B * T) * point_size

        return (temp_output,  # B * (F * temp_kernels) * T
                point_output)  # (B * T) * point_size


    def forward(self, X, diagnoses, flat, time_before_pred=5):

        # flat is B * no_flat_features
        # diagnoses is B * D
        # X is B * (2F + 2) * T
        # X_mask is B * T
        # (the batch is padded to the longest sequence, the + 2 is the time and the hour which are not for temporal convolution)

        # get rid of the time and hour fields - these shouldn't go through the temporal network
        # and split into features and indicator variables
        X_separated = torch.split(X[:, 1:-1, :], self.F, dim=1)  # tuple ((B * F * T), (B * F * T))

        # prepare repeat arguments and initialise layer loop
        B, _, T = X_separated[0].shape
        if self.model_type in ['pointwise_only', 'tpc']:
            repeat_flat = flat.repeat_interleave(T, dim=0)  # (B * T) * no_flat_features
            if self.no_mask:
                X_orig = cat((X_separated[0],
                              X[:, 0, :].unsqueeze(1),
                              X[:, -1, :].unsqueeze(1)), dim=1).permute(0, 2, 1).contiguous().view(B * T, self.F + 2)  # (B * T) * (F + 2)
            else:
                X_orig = X.permute(0, 2, 1).contiguous().view(B * T, 2 * self.F + 2)  # (B * T) * (2F + 2)
            repeat_args = {'repeat_flat': repeat_flat,
                           'X_orig': X_orig,
                           'B': B,
                           'T': T}
            if self.model_type == 'tpc':
                if self.no_mask:
                    next_X = X_separated[0]
                else:
                    next_X = torch.stack(X_separated, dim=2).reshape(B, 2 * self.F, T)  # B * 2F * T
                point_skip = X_separated[0]  # keeps track of skip connections generated from linear layers; B * F * T
                temp_output = None
                point_output = None
            else:  # pointwise only
                next_X = X_orig
                point_skip = None
        elif self.model_type == 'temp_only':
            next_X = torch.stack(X_separated, dim=2).view(B, 2 * self.F, T)  # B * 2F * T
            X_temp_orig = X_separated[0]  # skip connections for temp only model
            repeat_args = {'X_temp_orig': X_temp_orig,
                           'B': B,
                           'T': T}

        if self.no_skip_connections:
            temp_output = next_X
            point_output = cat((X_orig,  # (B * T) * (2F + 2)
                                repeat_flat),  # (B * T) * no_flat_features
                               dim=1)  # (B * T) * (2F + 2 + no_flat_features)
            self.layer1 = True

        for i in range(self.n_layers):
            kwargs = dict(self.layer_modules[str(i)], **repeat_args)
            if self.model_type == 'tpc':
                if self.no_skip_connections:
                    temp_output, point_output = self.temp_pointwise_no_skip(prev_point=point_output, prev_temp=temp_output,
                                                                            temp_kernels=self.layers[i]['temp_kernels'],
                                                                            padding=self.layers[i]['padding'], **kwargs)

                else:
                    temp_output, point_output, next_X, point_skip = self.temp_pointwise(X=next_X, point_skip=point_skip,
                                                                        prev_temp=temp_output, prev_point=point_output,
                                                                        temp_kernels=self.layers[i]['temp_kernels'],
                                                                        padding=self.layers[i]['padding'],
                                                                        point_size=self.layers[i]['point_size'],
                                                                        **kwargs)
            elif self.model_type == 'temp_only':
                next_X = self.temp(X=next_X, temp_kernels=self.layers[i]['temp_kernels'],
                                   padding=self.layers[i]['padding'], **kwargs)
            elif self.model_type == 'pointwise_only':
                next_X, point_skip = self.point(X=next_X, point_skip=point_skip, **kwargs)

        # tidy up
        if self.model_type == 'pointwise_only':
            next_X = next_X.view(B, T, -1).permute(0, 2, 1)
        elif self.no_skip_connections:
            # combine the final layer
            next_X = cat((point_output,
                          temp_output.permute(0, 2, 1).contiguous().view(B * T, self.F * self.layers[-1]['temp_kernels'])),
                         dim=1)
            next_X = next_X.view(B, T, -1).permute(0, 2, 1)

        # note that we cut off at time_before_pred hours here because the model is only valid from time_before_pred hours onwards
        if self.no_diag:
            combined_features = cat((flat.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * no_flat_features
                                     next_X[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)), dim=1)  # (B * (T - time_before_pred)) * (((F + Zt) * (1 + Y)) + no_flat_features) for tpc
        else:
            diagnoses_enc = self.relu(self.main_dropout(self.bn_diagnosis_encoder(self.diagnosis_encoder(diagnoses))))  # B * diagnosis_size
            combined_features = cat((flat.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * no_flat_features
                                     diagnoses_enc.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * diagnosis_size
                                     next_X[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)), dim=1)  # (B * (T - time_before_pred)) * (((F + Zt) * (1 + Y)) + diagnosis_size + no_flat_features) for tpc

        last_point_los = self.relu(self.main_dropout(self.bn_point_last_los(self.point_last_los(combined_features))))
        last_point_mort = self.relu(self.main_dropout(self.bn_point_last_mort(self.point_last_mort(combined_features))))

        if self.no_exp:
            los_predictions = self.hardtanh(self.point_final_los(last_point_los).view(B, T - time_before_pred))  # B * (T - time_before_pred)
        else:
            los_predictions = self.hardtanh(exp(self.point_final_los(last_point_los).view(B, T - time_before_pred)))  # B * (T - time_before_pred)
        mort_predictions = self.sigmoid(self.point_final_mort(last_point_mort).view(B, T - time_before_pred))  # B * (T - time_before_pred)

        return los_predictions, mort_predictions


    def temp_pointwise_no_skip_old(self, B=None, T=None, temp=None, bn_temp=None, point=None, bn_point=None, padding=None, prev_temp=None,
                               prev_point=None, temp_kernels=None, X_orig=None, repeat_flat=None):

        ### Temporal component ###

        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)
        # prev_temp shape: B * (F * Y) * T; N.B exception in the first layer where there are also mask features, in this case it is B * 2F * T

        X_padded = pad(prev_temp, padding, 'constant', 0)  # B * (F * Y) * (T + padding)
        temp_output = self.relu(self.temp_dropout(bn_temp(temp(X_padded))))  # B * (F * temp_kernels) * T

        ### Pointwise component ###

        # prev_point shape: (B * T) * ((F * Y) + Z)

        # if this is not layer 1:
        if self.layer1:
            X_concat = prev_point
            self.layer1 = False
        else:
            X_concat = cat((prev_point,
                            prev_temp.permute(0, 2, 1).contiguous().view(B * T, self.F * temp_kernels)),
                           dim=1)

        point_output = self.relu(self.main_dropout(bn_point(point(X_concat))))  # (B * T) * point_size

        return (temp_output,  # B * (F * temp_kernels) * T
                point_output)  # (B * T) * point_size


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