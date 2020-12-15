import pandas as pd


def preprocess_flat(flat):

    # make naming consistent with the other tables
    flat.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    flat.set_index('patient', inplace=True)

    # admission diagnosis is dealt with in diagnoses.py not flat features
    flat.drop(columns=['apacheadmissiondx'], inplace=True)

    # drop apache variables as these aren't available until 24 hours into the stay
    flat.drop(columns=['eyes', 'motor', 'verbal', 'dialysis', 'vent', 'meds', 'intubated', 'bedcount'], inplace=True)

    flat['gender'].replace({'Male': 1, 'Female': 0}, inplace=True)
    flat['teachingstatus'].replace({'t': 1, 'f': 0}, inplace=True)

    cat_features = ['ethnicity', 'unittype', 'unitadmitsource', 'unitvisitnumber', 'unitstaytype',
                                         'physicianspeciality', 'numbedscategory', 'region']
    # get rid of any really uncommon values
    for f in cat_features:
        too_rare = [value for value, count in flat[f].value_counts().iteritems() if count < 1000]
        flat.loc[flat[f].isin(too_rare), f] = 'misc'

    # convert the categorical features to one-hot
    flat = pd.get_dummies(flat, columns=cat_features)

    # 10 patients have NaN for age; we fill this with the mean value which is 63
    flat['age'].fillna('63', inplace=True)
    # some of the ages are like '> 89' rather than numbers, this needs removing and converting to numbers
    # but we make an extra variable to keep this information
    flat['> 89'] = flat['age'].str.contains('> 89').astype(int)
    flat['age'] = flat['age'].replace('> ', '', regex=True)
    flat['age'] = [float(value) for value in flat.age.values]

    # note that the features imported from the time series have already been normalised
    # standardisation is for features that are probably normally distributed
    features_for_standardisation = 'admissionheight'
    means = flat[features_for_standardisation].mean(axis=0)
    stds = flat[features_for_standardisation].std(axis=0)
    flat[features_for_standardisation] = (flat[features_for_standardisation] - means) / stds

    # probably not normally distributed
    features_for_min_max = ['admissionweight', 'age', 'hour']#, 'bedcount']

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
    # these are mainly found in admissionweight and admissionheight,
    # so we create another variable to tell the model when this has been imputed
    flat['nullweight'] = flat['admissionweight'].isnull().astype(int)
    flat['nullheight'] = flat['admissionheight'].isnull().astype(int)
    flat['admissionweight'].fillna(0, inplace=True)
    flat['admissionheight'].fillna(0, inplace=True)
    # there are only 11 missing genders but we might as well set this to 0.5 to tell the model we aren't sure
    flat['gender'].fillna(0.5, inplace=True)
    flat['gender'].replace({'Other': 0.5, 'Unknown': 0.5}, inplace=True)

    return flat

def preprocess_labels(labels):

    # make naming consistent with the other tables
    labels.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    labels.set_index('patient', inplace=True)

    labels = pd.get_dummies(labels, columns=['unitdischargelocation', 'unitdischargestatus'])

    labels['actualhospitalmortality'].replace({'EXPIRED': 1, 'ALIVE': 0}, inplace=True)

    return labels

def flat_and_labels_main(eICU_path):

    print('==> Loading data from labels and flat features files...')
    flat = pd.read_csv(eICU_path + 'flat_features.csv')
    flat = preprocess_flat(flat)
    flat.sort_index(inplace=True)
    labels = pd.read_csv(eICU_path + 'labels.csv')
    labels = preprocess_labels(labels)
    labels.sort_index(inplace=True)

    # filter out any patients that don't have timeseries
    try:
        with open(eICU_path + 'stays.txt', 'r') as f:
            ts_patients = [int(patient.rstrip()) for patient in f.readlines()]
    except FileNotFoundError:
        ts_patients = pd.read_csv(eICU_path + 'preprocessed_timeseries.csv')
        ts_patients = [x for x in ts_patients.patient.unique()]
        with open(eICU_path + 'stays.txt', 'w') as f:
            for patient in ts_patients:
                f.write("%s\n" % patient)
    flat = flat.loc[ts_patients].copy()
    labels = labels.loc[ts_patients].copy()

    print('==> Saving finalised preprocessed labels and flat features...')
    flat.to_csv(eICU_path + 'preprocessed_flat.csv')
    labels.to_csv(eICU_path + 'preprocessed_labels.csv')
    return

if __name__=='__main__':
    from eICU_preprocessing.run_all_preprocessing import eICU_path
    flat_and_labels_main(eICU_path)
