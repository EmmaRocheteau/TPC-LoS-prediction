import pandas as pd
import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

def print_metrics_mortality(df):
    mean_all, conf_bound = mean_confidence_interval(df)

    for i, (m, cb) in enumerate(zip(mean_all, conf_bound)):
        if i == 0:
            m = str(np.round(m, 3)).ljust(5, '0')
            cb = str(np.round(cb, 3)).ljust(5, '0')
        elif i == 1:
            m = str(np.round(m, 3)).ljust(5, '0')
            cb = str(np.round(cb, 3)).ljust(5, '0')
        and_space = ' & '
        print('{}$\pm${}{}'.format(m, cb, and_space), end='', flush=True)

def print_metrics_los(df):
    mean_all, conf_bound = mean_confidence_interval(df)
    mean_all = swapPositions(mean_all, 1, 2)
    conf_bound = swapPositions(conf_bound, 1, 2)

    for i, (m, cb) in enumerate(zip(mean_all, conf_bound)):
        if i == 0:
            m = str(np.round(m, 2)).ljust(4, '0')
            cb = str(np.round(cb, 2)).ljust(4, '0')
        elif i in [1, 2]:
            m = str(np.round(m, 1)).ljust(4, '0')
            cb = str(np.round(cb, 1)).ljust(3, '0')
        else:
            m = str(np.round(m, 2)).ljust(4, '0')
            cb = str(np.round(cb, 2)).ljust(4, '0')
        and_space = ' & ' if i < 5 else ' \\\\'
        print('{}$\pm${}{}'.format(m, cb, and_space), end='', flush=True)

def get_metrics(dataset, task, experiment):
    print('Experiment: {}'.format(experiment))
    stats = pd.read_csv('models/experiments/final/{}/{}/{}/results.csv'.format(dataset, task, experiment), header=None)
    print('There are {} experiments done here'.format(len(stats)))
    print('Discarding the first {}'.format(len(stats) - 10))
    stats = stats[-10:]
    if task == 'mortality':
        stats.columns = ['acc', 'prec0', 'prec1', 'rec0', 'rec1', 'auroc', 'auprc', 'f1macro']
    elif task == 'LoS':
        stats.columns = ['mad', 'mse', 'mape', 'msle', 'r2', 'kappa']
    elif task == 'multitask':
        stats.columns = ['mad', 'mse', 'mape', 'msle', 'r2', 'kappa', 'acc', 'prec0', 'prec1', 'rec0', 'rec1', 'auroc',
                         'auprc', 'f1macro']
    return stats

def main(dataset, task, experiment):
    stats = get_metrics(dataset, task, experiment)
    if task == 'mortality':
        print_metrics_mortality(stats[['auroc', 'auprc']])
        print('- & - & - & - & - & - \\\\')
    if task == 'LoS':
        print('- & - & ', end='')
        print_metrics_los(stats)
    if task == 'multitask':
        print_metrics_mortality(stats[['auroc', 'auprc']])
        print_metrics_los(stats[['mad', 'mse', 'mape', 'msle', 'r2', 'kappa']])
    return

if __name__=='__main__':
    dataset = 'MIMIC'
    task = 'mortality'
    experiment = 'ChannelwiseLSTM'
    main(dataset, task, experiment)