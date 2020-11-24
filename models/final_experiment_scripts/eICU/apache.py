import pandas as pd
from models.metrics import print_metrics_regression
from eICU_preprocessing.run_all_preprocessing import eICU_path

labels = pd.read_csv(eICU_path + 'test/labels.csv')
metrics_list = print_metrics_regression(labels.actualiculos, labels.predictediculos)
