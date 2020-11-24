import pandas as pd
from eICU_preprocessing.timeseries import reconfigure_timeseries, resample_and_mask, gen_patient_chunk, further_processing
import json
import os
import numpy as np


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

    # note that in MIMIC the timeseries are a lot messier so there are a lot of variables present that are not useful
    # drop duplicate columns which appear in chartevents
    print('==> Dropping the following columns because they have duplicates in labevents:')
    cols = []
    for col in timeseries.columns:
        if col in timeseries_lab.columns or col in timeseries_lab.columns + ' (serum)':
            cols.append(col)
    # plus some others which don't quite match up based on strings
    cols += ['WBC', 'HCO3 (serum)', 'Lactic Acid', 'PH (Arterial)', 'Arterial O2 pressure', 'Arterial CO2 Pressure',
             'Arterial Base Excess', 'TCO2 (calc) Arterial', 'Ionized Calcium', 'BUN', 'Calcium non-ionized', 'Anion gap']
    for col in cols:
        print('\t' + col)
    timeseries.drop(columns=cols, inplace=True)

    # just take a single Braden score, the individual variables will be deleted
    timeseries['Braden Score'] = timeseries[['Braden Activity', 'Braden Friction/Shear', 'Braden Mobility',
                                             'Braden Moisture', 'Braden Nutrition', 'Braden Sensory Perception']].sum(axis=1)
    timeseries['Braden Score'].replace(0, np.nan, inplace=True)  # this is where it hasn't been measured

    # finally remove some binary and less useful variables from the original set
    print('==> Also removing some binary and less useful variables:')
    other = ['18 Gauge Dressing Occlusive', '18 Gauge placed in outside facility', '18 Gauge placed in the field',
             '20 Gauge Dressing Occlusive', '20 Gauge placed in outside facility', '20 Gauge placed in the field',
             'Alarms On', 'Ambulatory aid', 'CAM-ICU MS Change', 'Eye Care', 'High risk (>51) interventions',
             'History of falling (within 3 mnths)', 'IV/Saline lock', 'Mental status', 'Parameters Checked',
             'ST Segment Monitoring On', 'Secondary diagnosis', 'Acuity Workload Question 1',
             'Acuity Workload Question 2', 'Arterial Line Dressing Occlusive', 'Arterial Line Zero/Calibrate',
             'Arterial Line placed in outside facility', 'Back Care', 'Cough/Deep Breath', 'Cuff Pressure',
             'Gait/Transferring', 'Glucose (whole blood)', 'Goal Richmond-RAS Scale', 'Inspiratory Ratio',
             'Inspiratory Time', 'Impaired Skin Odor #1', 'Braden Activity', 'Braden Friction/Shear', 'Braden Mobility',
             'Braden Moisture', 'Braden Nutrition', 'Braden Sensory Perception', 'Multi Lumen placed in outside facility',
             'O2 Saturation Pulseoxymetry Alarm - High', 'Orientation', 'Orientation to Person',
             'Orientation to Place', 'Orientation to Time', 'Potassium (whole blood)', 'Skin Care',
             'SpO2 Desat Limit', 'Subglottal Suctioning', 'Ventilator Tank #1', 'Ventilator Tank #2', 'Ventilator Type']
    for col in other:
        print('\t' + col)
    timeseries.drop(columns=other, inplace=True)

    #''' Code for deciding which variables to keep - nice with a breakpoint in the indicated position'''
    #import matplotlib.pyplot as plt
    #for col in timeseries.columns:
    #    plt.hist(timeseries[timeseries[col].notnull()][col])
    #    plt.show()
    #    print(col)
    #    break_point_here = None

    patients = timeseries.index.unique(level=0)

    size = 4000
    gen_chunks = gen_patient_chunk(patients, size=size)
    i = size
    header = True  # for the first chunk include the header in the csv file

    print('==> Starting main processing loop...')

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