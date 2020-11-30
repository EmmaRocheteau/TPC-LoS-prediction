import pandas as pd
from eICU_preprocessing.split_train_test import process_table, shuffle_stays


def shuffle_train(train_path):

    labels = pd.read_csv(train_path + '/labels.csv', index_col='patient')
    flat = pd.read_csv(train_path + '/flat.csv', index_col='patient')
    diagnoses = pd.read_csv(train_path + '/diagnoses.csv', index_col='patient')
    timeseries = pd.read_csv(train_path + '/timeseries.csv', index_col='patient')

    stays = labels.index.values
    stays = shuffle_stays(stays, seed=None)  # No seed will make it completely random
    for table_name, table in zip(['labels', 'flat', 'diagnoses', 'timeseries'],
                                 [labels, flat, diagnoses, timeseries]):
        process_table(table_name, table, stays, train_path)

    with open(train_path + '/stays.txt', 'w') as f:
        for stay in stays:
            f.write("%s\n" % stay)
    return