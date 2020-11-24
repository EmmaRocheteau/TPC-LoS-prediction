import pandas as pd


def preprocess_flat(flat):

    # make naming consistent with the other tables
    flat.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    flat.set_index('patient', inplace=True)

    flat['gender'].replace({'M': 1, 'F': 0}, inplace=True)

    cat_features = ['ethnicity', 'first_careunit', 'admission_location', 'insurance']
    # get rid of any really uncommon values
    for f in cat_features:
        too_rare = [value for value, count in flat[f].value_counts().iteritems() if count < 1000]
        flat.loc[flat[f].isin(too_rare), f] = 'misc'

    # convert the categorical features to one-hot
    flat = pd.get_dummies(flat, columns=cat_features)

    # note that the features imported from the time series have already been normalised
    # standardisation is for features that are probably normally distributed
    features_for_standardisation = 'height'
    means = flat[features_for_standardisation].mean(axis=0)
    stds = flat[features_for_standardisation].std(axis=0)
    flat[features_for_standardisation] = (flat[features_for_standardisation] - means) / stds

    # probably not normally distributed
    features_for_min_max = ['weight', 'age', 'hour', 'eyes', 'motor', 'verbal']

    def scale_min_max(flat):
        quantiles = flat.quantile([0.05, 0.95])
        maxs = quantiles.loc[0.95]
        mins = quantiles.loc[0.05]
        return 2 * (flat - mins) / (maxs - mins) - 1

    flat[features_for_min_max] = flat[features_for_min_max].apply(scale_min_max)

    # we then need to make sure that ridiculous outliers are clipped to something sensible
    flat[features_for_standardisation] = flat[features_for_standardisation].clip(lower=-4, upper=4)  # room for +- 3 on each side of the normal range, as variables are scaled roughly between -1 and 1
    flat[features_for_min_max] = flat[features_for_min_max].clip(lower=-4, upper=4)

    # fill in the NaNs
    # these are mainly found in height
    # so we create another variable to tell the model when this has been imputed
    flat['nullheight'] = flat['height'].isnull().astype(int)
    flat['weight'].fillna(0, inplace=True)  # null in only 83 patients
    flat['height'].fillna(0, inplace=True)  # null in 38217 patients
    flat['eyes'].fillna(0, inplace=True)  # null in 192 patients
    flat['motor'].fillna(0, inplace=True)  # null in 270 patients
    flat['verbal'].fillna(0, inplace=True)  # null in 6240 patients

    return flat

def preprocess_labels(labels):

    # make naming consistent with the other tables
    labels.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    labels.set_index('patient', inplace=True)

    return labels

def flat_and_labels_main(MIMIC_path):

    print('==> Loading data from labels and flat features files...')
    flat = pd.read_csv(MIMIC_path + 'flat_features.csv')
    flat = preprocess_flat(flat)
    flat.sort_index(inplace=True)
    labels = pd.read_csv(MIMIC_path + 'labels.csv')
    labels = preprocess_labels(labels)
    labels.sort_index(inplace=True)

    # filter out any patients that don't have timeseries
    try:
        with open(MIMIC_path + 'stays.txt', 'r') as f:
            ts_patients = [int(patient.rstrip()) for patient in f.readlines()]
    except FileNotFoundError:
        ts_patients = pd.read_csv(MIMIC_path + 'preprocessed_timeseries.csv')
        ts_patients = [x for x in ts_patients.patient.unique()]
        with open(MIMIC_path + 'stays.txt', 'w') as f:
            for patient in ts_patients:
                f.write("%s\n" % patient)
    flat = flat.loc[ts_patients].copy()
    labels = labels.loc[ts_patients].copy()

    print('==> Saving finalised preprocessed labels and flat features...')
    flat.to_csv(MIMIC_path + 'preprocessed_flat.csv')
    labels.to_csv(MIMIC_path + 'preprocessed_labels.csv')
    return

if __name__=='__main__':
    from MIMIC_preprocessing.run_all_preprocessing import MIMIC_path
    flat_and_labels_main(MIMIC_path)