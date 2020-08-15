import pandas as pd
import numpy as np
from interpretability.summary_plot import summary_plot
import matplotlib.pyplot as plt


def attr_plot(feature_type, plot_size):

    if feature_type == 'diagnoses':
        shorthand = 'diag'
        color_bar = False
    elif feature_type == 'timeseries':
        shorthand = 'ts'
        color_bar = True
    elif feature_type == 'abs_timeseries':
        shorthand = 'abs_ts'
        feature_type = 'timeseries'
        color_bar = True
    else:
        shorthand = feature_type
        color_bar = True

    features = pd.read_csv('interpretability/{}_fts.csv'.format(shorthand), header=None).values
    feature_names = list(pd.read_csv('eICU_data/test/{}.csv'.format(feature_type), nrows=0).columns)
    feature_names.pop(0)
    attr = pd.read_csv('interpretability/{}_attr.csv'.format(shorthand), header=None).values
    attr = attr.clip(min=-20, max=20)
    # features with zero attribution
    nonzero_attr = np.nansum(attr, axis=0).nonzero()[0]
    if feature_type == 'timeseries':
        # only keep the non mask
        nonzero_attr = nonzero_attr[nonzero_attr < 88]
    attr = attr[:, nonzero_attr]
    features = features[:, nonzero_attr]
    feature_names = [feature_names[i] for i in nonzero_attr]
    summary_plot(attr, features=features, feature_names=feature_names, show=False, plot_size=plot_size,
                 plot_type='dot', sort=True, max_display=25, color_bar=color_bar)
    plt.tight_layout()
    plt.xlabel('Integrated Gradient')
    plt.savefig('interpretability/{}_plot.png'.format(shorthand), dpi=300)

    plt.show()

    return

attr_plot('diagnoses', (14, 7))
attr_plot('abs_timeseries', (9, 7))
attr_plot('flat', (8, 7))