import pandas as pd
from shap import summary_plot
import matplotlib.pyplot as plt

feature_type = 'ts'

features = pd.read_csv('interpretability/ts_fts.csv', header=None).values
feature_names = list(pd.read_csv('eICU_data/preprocessed_timeseries.csv', nrows=0).columns)
feature_names.pop(0)
flat_weights = pd.read_csv('interpretability/ts_attr.csv', header=None).values
flat_weights = flat_weights.clip(min=-15, max=15)
#features = features[:5000, :]
#flat_weights = flat_weights[:5000, :]
summary_plot(flat_weights, features=features, feature_names=feature_names, show=False, plot_size=(10, 7))
plt.tight_layout()
plt.xlabel('Integrated Gradient')
plt.savefig('interpretability/ts_plot.png', dpi=300)

