def best_global(c):
    c['alpha'] = 100
    if c['dataset'] == 'eICU':
        c['main_dropout_rate'] = 0.45
        c['last_linear_size'] = 17
        c['diagnosis_size'] = 64
        c['batch_norm'] = 'mybatchnorm'
    elif c['dataset'] == 'MIMIC':
        # diagnosis size does not apply for MIMIC since we don't have diagnoses
        c['main_dropout_rate'] = 0
        c['last_linear_size'] = 36
        c['batch_norm'] = 'mybatchnorm'
    return c

def best_tpc(c):
    c = best_global(c)
    c['mode'] = 'test'
    c['model_type'] = 'tpc'
    if c['dataset'] == 'eICU':
        if c['percentage_data'] == 6.25:
            c['n_epochs'] = 8
        elif c['task'] == 'mortality':
            c['n_epochs'] = 6
        else:
            c['n_epochs'] = 15
        c['batch_size'] = 32
        c['n_layers'] = 9
        c['kernel_size'] = 4
        c['no_temp_kernels'] = 12
        c['point_size'] = 13
        c['learning_rate'] = 0.00226
        c['temp_dropout_rate'] = 0.05
        c['temp_kernels'] = [12] * 9 if not c['share_weights'] else [32] * 9
        c['point_sizes'] = [13] * 9
    elif c['dataset'] == 'MIMIC':
        c['no_diag'] = True
        c['n_epochs'] = 10 if c['task'] is not 'mortality' else 6
        c['batch_size'] = 8
        c['batch_size_test'] = 8  # purely to keep experiment size small so I can run many in parallel
        c['n_layers'] = 8
        c['kernel_size'] = 5
        c['no_temp_kernels'] = 11
        c['point_size'] = 5
        c['learning_rate'] = 0.00221
        c['temp_dropout_rate'] = 0.05
        c['temp_kernels'] = [11] * 8
        c['point_sizes'] = [5] * 8
    return c

def best_lstm(c):
    c = best_global(c)
    c['mode'] = 'test'
    if c['dataset'] == 'eICU':
        c['batch_size'] = 512
        c['n_layers'] = 2
        c['hidden_size'] = 128
        c['learning_rate'] = 0.00129
        c['lstm_dropout_rate'] = 0.2
        if c['percentage_data'] < 25:
            c['n_epochs'] = 4
        elif c['percentage_data'] == 25:
            c['n_epochs'] = 5
        elif c['percentage_data'] == 50:
            c['n_epochs'] = 6
        else:
            c['n_epochs'] = 8
    elif c['dataset'] == 'MIMIC':
        c['no_diag'] = True
        c['batch_size'] = 32
        c['n_layers'] = 1
        c['hidden_size'] = 128
        c['learning_rate'] = 0.00163
        c['lstm_dropout_rate'] = 0.25
        c['n_epochs'] = 8
    return c

def best_cw_lstm(c):
    c['mode'] = 'test'
    c['channelwise'] = True
    # carry over the best parameters from lstm, including global
    c = best_lstm(c)
    if c['dataset'] == 'eICU':
        c['hidden_size'] = 8
        if c['percentage_data'] < 25:
            c['n_epochs'] = 15
        elif c['percentage_data'] == 25 or c['task'] == 'mortality':
            c['n_epochs'] = 20
        elif c['percentage_data'] == 50:
            c['n_epochs'] = 25
        else:
            c['n_epochs'] = 30
    elif c['dataset'] == 'MIMIC':
        c['no_diag'] = True
        c['hidden_size'] = 8
        c['n_epochs'] = 20
    return c


def best_transformer(c):
    c = best_global(c)
    c['mode'] = 'test'
    if c['dataset'] == 'eICU':
        c['batch_size'] = 32
        c['n_layers'] = 6
        c['feedforward_size'] = 256
        c['d_model'] = 16
        c['n_heads'] = 2
        c['learning_rate'] = 0.00017
        c['trans_dropout_rate'] = 0
        if c['percentage_data'] < 12.5:
            c['n_epochs'] = 8
        elif c['percentage_data'] == 12.5:
            c['n_epochs'] = 10
        elif c['percentage_data'] == 25:
            c['n_epochs'] = 12
        elif c['percentage_data'] == 50:
            c['n_epochs'] = 14
        else:
            c['n_epochs'] = 15
    elif c['dataset'] == 'MIMIC':
        c['no_diag'] = True
        c['batch_size'] = 64
        c['n_layers'] = 2
        c['feedforward_size'] = 64
        c['d_model'] = 32
        c['n_heads'] = 1
        c['learning_rate'] = 0.00129
        c['trans_dropout_rate'] = 0.05
        c['n_epochs'] = 15
    return c