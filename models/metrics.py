from sklearn import metrics
import numpy as np

class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)

def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0]
        b = CustomBins.bins[i][1]
        if a <= x < b:
            if one_hot:
                onehot = np.zeros((CustomBins.nbins,))
                onehot[i] = 1
                return onehot
            return i
    return None

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(4/24, y_true))) * 100  # this stops the mape being a stupidly large value when y_true happens to be very small

def mean_squared_logarithmic_error(y_true, y_pred):
    return np.mean(np.square(np.log(y_true/y_pred)))

def print_metrics_regression(y_true, predictions, verbose=1, elog=None):
    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if elog is not None:
        elog.print('Custom bins confusion matrix:')
        elog.print(cf)
    elif verbose:
        print('Custom bins confusion matrix:')
        print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins, weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)
    msle = mean_squared_logarithmic_error(y_true, predictions)
    r2 = metrics.r2_score(y_true, predictions)

    if verbose:
        print('Mean absolute deviation (MAD) = {}'.format(mad))
        print('Mean squared error (MSE) = {}'.format(mse))
        print('Mean absolute percentage error (MAPE) = {}'.format(mape))
        print('Mean squared logarithmic error (MSLE) = {}'.format(msle))
        print('R^2 Score = {}'.format(r2))
        print('Cohen kappa score = {}'.format(kappa))

    return [mad, mse, mape, msle, r2, kappa]