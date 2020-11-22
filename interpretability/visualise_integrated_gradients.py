import pandas as pd
import numpy as np
from shap import summary_plot
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def attr_plot(feature_type, plot_size):

    if feature_type == 'diagnoses':
        shorthand = 'diag'
        color_bar = False
        xlim = (-1.5, 1.5)
        xticks = ['-1.5-', '-1.0', '-0.5', '0', '0.5', '1.0', '1.5+']
        xlabels = [-1.5, -1.0, -0.5, 0, 0.5, 1, 1.5]
    elif feature_type == 'timeseries':
        shorthand = 'ts'
        color_bar = True
        xlim = (-3, 3)
        xticks = ['-3-', '-2', '-1', '0', '1', '2', '3+']
        xlabels = [-3, -2, -1, 0, 1, 2, 3]
    elif feature_type == 'abs_timeseries':
        shorthand = 'abs_ts'
        feature_type = 'timeseries'
        color_bar = True
        xlim = (0, 2.5)
        xticks = ['0', '0.5', '1.0', '1.5', '2.0', '2.5+']
        xlabels = [0, 0.5, 1, 1.5, 2, 2.5]
    else:
        shorthand = feature_type
        color_bar = True
        xlim = (-3, 3)
        xticks = ['-3-', '-2', '-1', '0', '1', '2', '3+']
        xlabels = [-3, -2, -1, 0, 1, 2, 3]
    features = pd.read_csv('interpretability/{}_fts48.csv'.format(shorthand), header=None).values
    feature_names = list(pd.read_csv('eICU_data/test/{}.csv'.format(feature_type), nrows=0).columns)
    feature_names.pop(0)
    attr = pd.read_csv('interpretability/{}_attr48.csv'.format(shorthand), header=None).values
    attr = attr.clip(min=xlim[0], max=xlim[1])
    # features with zero attribution
    nonzero_attr = np.nansum(attr, axis=0).nonzero()[0]
    if feature_type == 'timeseries':
        # only keep the non mask
        nonzero_attr = nonzero_attr[nonzero_attr < 88]
    attr = attr[:, nonzero_attr]
    features = features[:, nonzero_attr]
    feature_names = [feature_names[i] for i in nonzero_attr]
    features = np.nan_to_num(features, 0)
    attr = np.nan_to_num(attr, 0)
    summary_plot(attr, features=features, feature_names=feature_names, show=False, plot_size=plot_size,
                 plot_type='bar', sort=True, max_display=25, color_bar=color_bar)
    plt.tight_layout()
    plt.xlabel('Average Integrated Gradient')
    #plt.xticks(xlabels, xticks)
    plt.savefig('figures/{}_plot48.png'.format(shorthand), dpi=300)
    plt.show()

    return

# diag not important for model so attributions meaningless
# timeseries too noisy so we take the mean absolute attribution over the whole timeseries
attr_plot('abs_timeseries', (10, 7))