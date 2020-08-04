import pandas as pd
import matplotlib.pyplot as plt

flat_fts = list(pd.read_csv('eICU_data/preprocessed_flat.csv', nrows=1))
flat_weights = pd.read_csv('interpretability/flat.csv').values
plt.plot(flat_fts, flat_weights)
plt.show()