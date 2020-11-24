from models.metrics import print_metrics_regression
from models.mean_median_model import mean_median

if __name__=='__main__':
    mean_train, median_train, test_y = mean_median(dataset='MIMIC')
    print('Total predictions:')
    print('Using mean value of {}...'.format(mean_train))
    metrics_list = print_metrics_regression(test_y['true'], test_y['mean'])
    print('Using median value of {}...'.format(median_train))
    metrics_list = print_metrics_regression(test_y['true'], test_y['median'])