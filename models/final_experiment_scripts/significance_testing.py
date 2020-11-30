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
    return

if __name__=='__main__':
    dataset = 'eICU'
    exp1 = get_metrics(dataset, 'LoS', 'StandardLSTM')
    exp2 = get_metrics(dataset, 'multitask', 'StandardLSTM')

    for param in exp1.columns:
        ttest(exp1[param], exp2[param], param)