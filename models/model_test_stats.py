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

exp_dir = 'StandardLSTMLabsOnly'

print('Experiment: {}'.format(exp_dir))

stats = pd.read_csv('models/experiments/final/' + exp_dir + '/results.csv', header=None)
stats.columns =['mad', 'mse', 'mape', 'msle', 'r2', 'kappa']
print('There are {} experiments done here'.format(len(stats)))
print('Discarding the first {}'.format(len(stats) - 10))
stats = stats[-10:]

mean_all, conf_bound = mean_confidence_interval(stats)
mean_all = swapPositions(mean_all, 1, 2)
conf_bound = swapPositions(conf_bound, 1, 2)

for i, (m, cb) in enumerate(zip(mean_all, conf_bound)):
    if i == 0:
        m = str(np.round(m, 2)).ljust(4, '0')
        cb = str(np.round(cb, 2)).ljust(4, '0')
    elif i in [1, 2]:
        m = str(np.round(m, 1)).ljust(4, '0')
        cb = str(np.round(cb, 1)).ljust(3, '0')
#    else:
#        m = str(np.round(m, 3)).ljust(5, '0')
#        cb = str(np.round(cb, 3)).ljust(5, '0')
    else:
        m = str(np.round(m, 2)).ljust(4, '0')
        cb = str(np.round(cb, 2)).ljust(4, '0')
    and_space = ' & ' if i < 5 else ' \\\\'
    print('\\footnotesize{}{}$\pm${}{}{}'.format('{', m, cb, '}', and_space), end='', flush=True)
