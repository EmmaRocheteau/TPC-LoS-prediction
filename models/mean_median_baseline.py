import pandas as pd
import numpy as np
import torch
from eICU_preprocessing.reader import eICUReader
from models.metrics import print_metrics_regression
from models.experiment_template import remove_padding
from eICU_preprocessing.run_all_preprocessing import eICU_path

device = torch.device('cpu')
train_datareader = eICUReader(eICU_path + 'train', device=device)
test_datareader = eICUReader(eICU_path + 'test', device=device)
train_batches = train_datareader.batch_gen(batch_size=512)
test_batches = test_datareader.batch_gen(batch_size=512)
bool_type = torch.cuda.BoolTensor if device == torch.device('cuda') else torch.BoolTensor
train_y = np.array([])
test_y = np.array([])

for batch_idx, (padded, mask, diagnoses, flat, labels, seq_lengths) in enumerate(train_batches):
    train_y = np.append(train_y, remove_padding(labels, mask.type(bool_type), device))

train_y = pd.DataFrame(train_y, columns=['true'])

mean_train = train_y.mean().values[0]
median_train = train_y.median().values[0]

for batch_idx, (padded, mask, diagnoses, flat, labels, seq_lengths) in enumerate(test_batches):
    test_y = np.append(test_y, remove_padding(labels, mask.type(bool_type), device))

test_y = pd.DataFrame(test_y, columns=['true'])

test_y['mean'] = mean_train
test_y['median'] = median_train

print('Total predictions:')
print('Using mean value of {}...'.format(mean_train))
metrics_list = print_metrics_regression(test_y['true'], test_y['mean'])
print('Using median value of {}...'.format(median_train))
metrics_list = print_metrics_regression(test_y['true'], test_y['median'])