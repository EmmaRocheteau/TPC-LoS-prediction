import pandas as pd
from shap import summary_plot
import matplotlib.pyplot as plt

preproc_features = pd.read_csv('eICU_data/preprocessed_diagnoses.csv')
feature_names = list(preproc_features.columns.values)
feature_names.pop(0)
features = preproc_features.drop(columns='patient').values
flat_weights = pd.read_csv('interpretability/diag.csv').values
flat_weights = flat_weights.clip(min=-4, max=4)
features = features[:5000, :]
flat_weights = flat_weights[:5000, :]
summary_plot(flat_weights, features=features, feature_names=feature_names, show=False)
plt.savefig('interpretability/diag_plot.png', dpi=300)

