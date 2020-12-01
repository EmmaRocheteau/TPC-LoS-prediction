import pandas as pd
import numpy as np
import torch
from eICU_preprocessing.reader import eICUReader
from MIMIC_preprocessing.reader import MIMICReader
from models.experiment_template import remove_padding
from eICU_preprocessing.run_all_preprocessing import eICU_path
from MIMIC_preprocessing.run_all_preprocessing import MIMIC_path
from models.metrics import print_metrics_regression


def mean_median(dataset):

    device = torch.device('cpu')
    if dataset == 'MIMIC':
        path = MIMIC_path
        reader = MIMICReader
    else:
        path = eICU_path
        reader = eICUReader
    train_datareader = reader(path + 'train', device=device)
    test_datareader = reader(path + 'test', device=device)
    train_batches = train_datareader.batch_gen(batch_size=512)
    test_batches = test_datareader.batch_gen(batch_size=512)
    bool_type = torch.cuda.BoolTensor if device == torch.device('cuda') else torch.BoolTensor
    train_y = np.array([])
    test_y = np.array([])

    for batch_idx, batch in enumerate(train_batches):

        # unpack batch
        if dataset == 'MIMIC':
            padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
        else:
            padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch

        train_y = np.append(train_y, remove_padding(los_labels, mask.type(bool_type), device))

    train_y = pd.DataFrame(train_y, columns=['true'])

    mean_train = train_y.mean().values[0]
    median_train = train_y.median().values[0]

    for batch_idx, batch in enumerate(test_batches):

        # unpack batch
        if dataset == 'MIMIC':
            padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
        else:
            padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch

        test_y = np.append(test_y, remove_padding(los_labels, mask.type(bool_type), device))

    test_y = pd.DataFrame(test_y, columns=['true'])

    test_y['mean'] = mean_train
    test_y['median'] = median_train

    print('Total predictions:')
    print('Using mean value of {}...'.format(mean_train))
    print_metrics_regression(test_y['true'], test_y['mean'])
    print('Using median value of {}...'.format(median_train))
    print_metrics_regression(test_y['true'], test_y['median'])

    return