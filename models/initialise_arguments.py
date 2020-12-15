from trixi.util import Config
import argparse


# all default values stated here are the best hyperparameters in the eICU dataset, not MIMIC
def initialise_arguments():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--dataset', default='eICU', type=str)
    parser.add_argument('-disable_cuda', action='store_true')
    parser.add_argument('-intermediate_reporting', action='store_true')
    parser.add_argument('--batch_size_test', default=32, type=int)
    parser.add_argument('-shuffle_train', action='store_true')
    parser.add_argument('-save_results_csv', action='store_true')
    parser.add_argument('--percentage_data', default=100.0, type=float)
    parser.add_argument('--task', default='LoS', type=str, help='can be either LoS, mortality, or multitask (both)')
    parser.add_argument('--mode', default='train', type=str, help='can be either train, which reports intermediate '
                                                                  'results on the training and validation sets each '
                                                                  'epoch, or test, which just runs all the epochs and '
                                                                  'only reports on the test set')

    # loss
    parser.add_argument('--loss', default='msle', type=str, help='can either be msle or mse')
    parser.add_argument('-sum_losses', action='store_false')  # keep this as true

    # ablations
    parser.add_argument('-labs_only', action='store_true')
    parser.add_argument('-no_mask', action='store_true')
    parser.add_argument('-no_diag', action='store_true')
    parser.add_argument('-no_labs', action='store_true')
    parser.add_argument('-no_exp', action='store_true')

    # shared hyper-parameters
    parser.add_argument('--alpha', default=100, type=int)  # for multitask
    parser.add_argument('--main_dropout_rate', default=0.45, type=float)
    parser.add_argument('--L2_regularisation', default=0, type=float)
    parser.add_argument('--last_linear_size', default=17, type=int)
    parser.add_argument('--diagnosis_size', default=64, type=int)
    parser.add_argument('--batchnorm', default='mybatchnorm', type=str, help='can be: none, pointwiseonly, temponly, '
                        'default, mybatchnorm or low_momentum. \nfconly, convonly and low_momentum are implemented with '
                        'mybatchnorm rather than default pytorch')
    return parser

def gen_config(parser):
    args = parser.parse_args()
    # prepare config dictionary, add all arguments from args
    c = Config()
    for arg in vars(args):
        c[arg] = getattr(args, arg)
    return c

def initialise_tpc_arguments():
    parser = initialise_arguments()
    parser.add_argument('--n_epochs', default=15, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_layers', default=9, type=int)
    parser.add_argument('--kernel_size', default=4, type=int)
    parser.add_argument('--no_temp_kernels', default=12, type=int)
    parser.add_argument('--point_size', default=13, type=int)
    parser.add_argument('--learning_rate', default=0.00226, type=float)
    parser.add_argument('--temp_dropout_rate', default=0.05, type=float)
    parser.add_argument('-share_weights', action='store_true')
    parser.add_argument('-no_skip_connections', action='store_true')
    c = gen_config(parser)
    c['temp_kernels'] = [c['no_temp_kernels']]*c['n_layers']
    c['point_sizes'] = [c['point_size']]*c['n_layers']
    if c['dataset'] == 'MIMIC':  # set no_diag to True if the dataset is MIMIC
        c['no_diag'] = True
    return c

def initialise_lstm_arguments():
    parser = initialise_arguments()
    parser.add_argument('--n_epochs', default=8, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--n_layers', default=2, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=0.00129, type=float)
    parser.add_argument('--lstm_dropout_rate', default=0.2, type=float)
    parser.add_argument('-bidirectional', action='store_true')
    parser.add_argument('-channelwise', action='store_true')
    c = gen_config(parser)
    if c['dataset'] == 'MIMIC':  # set no_diag to True if the dataset is MIMIC
        c['no_diag'] = True
    return c

def initialise_transformer_arguments():
    parser = initialise_arguments()
    parser.add_argument('--n_epochs', default=15, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_layers', default=6, type=int)
    parser.add_argument('--feedforward_size', default=256, type=int)
    parser.add_argument('--d_model', default=16, type=int)
    parser.add_argument('--n_heads', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.00017, type=float)
    parser.add_argument('--trans_dropout_rate', default=0, type=float)
    parser.add_argument('-positional_encoding', action='store_true')  # default is False
    c = gen_config(parser)
    if c['dataset'] == 'MIMIC':  # set no_diag to True if the dataset is MIMIC
        c['no_diag'] = True
    return c