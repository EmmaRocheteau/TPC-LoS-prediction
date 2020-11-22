import pandas as pd
from eICU_preprocessing.timeseries import reconfigure_timeseries, resample_and_mask, gen_patient_chunk, further_processing
import json
import os

def gen_timeseries_file(MIMIC_path, test=False):

    print('==> Loading data from timeseries files...')
    if test:
        timeseries_lab = pd.read_csv(MIMIC_path + 'timeserieslab.csv', nrows=500000)
        timeseries = pd.read_csv(MIMIC_path + 'timeseries.csv', nrows=500000)
    else:
        timeseries_lab = pd.read_csv(MIMIC_path + 'timeserieslab.csv')
        timeseries = pd.read_csv(MIMIC_path + 'timeseries.csv')

    print('==> Reconfiguring lab timeseries...')
    timeseries_lab = reconfigure_timeseries(timeseries_lab,
                                            offset_column='labresultoffset',
                                            feature_column='labname',
                                            test=test)
    timeseries_lab.columns = timeseries_lab.columns.droplevel()

    print('==> Reconfiguring other timeseries...')
    timeseries = reconfigure_timeseries(timeseries,
                                        offset_column='chartoffset',
                                        feature_column='chartvaluelabel',
                                        test=test)
    timeseries.columns = timeseries.columns.droplevel()

    patients = timeseries.index.unique(level=0)

    size = 4000
    gen_chunks = gen_patient_chunk(patients, size=size)
    i = size
    header = True  # for the first chunk include the header in the csv file

    for patient_chunk in gen_chunks:

        merged = timeseries_lab.loc[patient_chunk].append(timeseries.loc[patient_chunk], sort=False)

        if i == size:  # fixed from first run
            # all if not all are not normally distributed
            quantiles = merged.quantile([0.05, 0.95])
            maxs = quantiles.loc[0.95]
            mins = quantiles.loc[0.05]

        merged = 2 * (merged - mins) / (maxs - mins) - 1

        # we then need to make sure that ridiculous outliers are clipped to something sensible
        merged.clip(lower=-4, upper=4, inplace=True)  # room for +- 3 on each side, as variables are scaled roughly between 0 and 1

        resample_and_mask(merged, MIMIC_path, header, mask_decay=True, decay_rate=4/3, test=test, verbose=False)
        print('==> Processed ' + str(i) + ' patients...')
        i += size
        header = False

    return

def timeseries_main(MIMIC_path, test=False):
    # make sure the preprocessed_timeseries.csv file is not there because the first section of this script appends to it
    if test is False:
        print('==> Removing the preprocessed_timeseries.csv file if it exists...')
        try:
            os.remove(MIMIC_path + 'preprocessed_timeseries.csv')
        except FileNotFoundError:
            pass
    gen_timeseries_file(MIMIC_path, test)
    further_processing(MIMIC_path, test)
    return

if __name__=='__main__':
    with open('paths.json', 'r') as f:
        MIMIC_path = json.load(f)["MIMIC_path"]
    test = True
    timeseries_main(MIMIC_path, test)