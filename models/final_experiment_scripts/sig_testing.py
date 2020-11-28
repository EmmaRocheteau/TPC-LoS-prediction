from scipy import stats
import pandas as pd

def sig_ttest(list_results1, list_results2):
    t2, p2 = stats.ttest_ind(list_results1, list_results2)
    print("p = " + str(p2))
    stars = ''
    if p2 < 0.05:
        stars +='*'
        if p2 < 0.01:
            stars += '*'
            if p2 < 0.001:
                stars += '*'
        print(stars)
    return

def get_metrics_los(exp_dir):
    print('Experiment: {}'.format(exp_dir))
    stats = pd.read_csv('models/experiments/final/' + exp_dir + '/results.csv', header=None)
    stats.columns =['mad', 'mse', 'mape', 'msle', 'r2', 'kappa']
    print('There are {} experiments done here'.format(len(stats)))
    print('Discarding the first {}'.format(len(stats) - 10))
    stats = stats[-10:]
    return stats

if __name__=='__main__':
    path = '/models/experiments/final/'
    exp1 = get_metrics_los('TPC')
    exp2 = get_metrics_los('TPCNoDecay')

    for param in exp1.columns:
        print(param)
        sig_ttest(exp1[param], exp2[param])