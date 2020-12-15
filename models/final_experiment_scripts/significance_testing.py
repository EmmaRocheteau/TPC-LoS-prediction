import numpy as np
from scipy import stats
from models.final_experiment_scripts.latex_table_stats import get_metrics


def ttest(list_results1, list_results2, param):
    t2, p2 = stats.ttest_ind(list_results1, list_results2)
    stars = ''
    if p2 < 0.05:
        stars +='*'
        if p2 < 0.01:
            stars += '*'
            if p2 < 0.001:
                stars += '*'
    if len(param) < 4:
        param += '  '
    print('{}\tp = {}\t{}'.format(param, str(p2), stars))

def perc_diff(list_results1, list_results2, param):
    mean1 = np.mean(list_results1)
    mean2 = np.mean(list_results2)
    if param in ['mad', 'mape', 'mse', 'msle']:  # metrics that have a best value of 0
        perc_improvement = (mean1 - mean2)/mean1 * 100
    elif param in ['r2', 'kappa']:  # metrics that have a best value of 1
        perc_improvement = (mean2 - mean1)/(1 - mean1) * 100
    if len(param) < 4:
        param += '  '
    print('{}\t {}'.format(param, str(perc_improvement)))


if __name__=='__main__':
    dataset = 'MIMIC'
    task1 = 'LoS'
    task2 = 'LoS'
    model1 = 'Transformer'
    model2 = 'TPC'
    # if comparing with multitask, put multitask second
    exp1 = get_metrics(dataset, task1, model1)
    exp2 = get_metrics(dataset, task2, model2)

    for param in exp1.columns:
        ttest(exp1[param], exp2[param], param)

    if task1 == task2 == 'LoS':
        for param in exp1.columns:  # with respect to the first experiment
            perc_diff(exp1[param], exp2[param], param)