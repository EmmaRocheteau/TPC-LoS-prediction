"""Permutation importance for estimators <- code mainly taken from sklearn"""
import numpy as np
from joblib import Parallel
from joblib import delayed
from sklearn.metrics import check_scoring
from sklearn.utils import Bunch
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils.validation import _deprecate_positional_args
from models.run_tpc import TPC
import numpy as np
import pandas as pd
from models.initialise_arguments import initialise_tpc_arguments
from pathlib import Path
from captum.attr import IntegratedGradients
import matplotlib as mpl
import matplotlib.pyplot as plt


def _calculate_permutation_scores(estimator, X, y, col_idx, random_state,
                                  n_repeats, scorer):
    """Calculate score when `col_idx` is permuted."""
    random_state = check_random_state(random_state)

    # Work on a copy of X to to ensure thread-safety in case of threading based
    # parallelism. Furthermore, making a copy is also useful when the joblib
    # backend is 'loky' (default) or the old 'multiprocessing': in those cases,
    # if X is large it will be automatically be backed by a readonly memory map
    # (memmap). X.copy() on the other hand is always guaranteed to return a
    # writable data-structure whose columns can be shuffled inplace.
    X_permuted = X.copy()
    scores = np.zeros(n_repeats)
    shuffling_idx = np.arange(X.shape[0])
    for n_round in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        if hasattr(X_permuted, "iloc"):
            col = X_permuted.iloc[shuffling_idx, col_idx]
            col.index = X_permuted.index
            X_permuted.iloc[:, col_idx] = col
        else:
            X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]
        feature_score = scorer(estimator, X_permuted, y)
        scores[n_round] = feature_score

    return scores

@_deprecate_positional_args
def permutation_importance(estimator, X, y, *, scoring=None, n_repeats=5,
                           n_jobs=None, random_state=None):
    """Permutation importance for feature evaluation [BRE]_.

    The :term:`estimator` is required to be a fitted estimator. `X` can be the
    data set used to train the estimator or a hold-out set. The permutation
    importance of a feature is calculated as follows. First, a baseline metric,
    defined by :term:`scoring`, is evaluated on a (potentially different)
    dataset defined by the `X`. Next, a feature column from the validation set
    is permuted and the metric is evaluated again. The permutation importance
    is defined to be the difference between the baseline metric and metric from
    permutating the feature column.

    Read more in the :ref:`User Guide <permutation_importance>`.

    Parameters
    ----------
    estimator : object
        An estimator that has already been :term:`fitted` and is compatible
        with :term:`scorer`.

    X : ndarray or DataFrame, shape (n_samples, n_features)
        Data on which permutation importance will be computed.

    y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
        Targets for supervised or `None` for unsupervised.

    scoring : string, callable or None, default=None
        Scorer to use. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer is used.

    n_repeats : int, default=5
        Number of times to permute a feature.

    n_jobs : int or None, default=None
        The number of jobs to use for the computation.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, default=None
        Pseudo-random number generator to control the permutations of each
        feature.
        Pass an int to get reproducible results across function calls.
        See :term: `Glossary <random_state>`.

    Returns
    -------
    result : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        importances_mean : ndarray, shape (n_features, )
            Mean of feature importance over `n_repeats`.
        importances_std : ndarray, shape (n_features, )
            Standard deviation over `n_repeats`.
        importances : ndarray, shape (n_features, n_repeats)
            Raw permutation importance scores.

    References
    ----------
    .. [BRE] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
             2001. https://doi.org/10.1023/A:1010933404324"""

    if not hasattr(X, "iloc"):
        X = check_array(X, force_all_finite='allow-nan', dtype=None)

    # Precompute random seed from the random state to be used
    # to get a fresh independent RandomState instance for each
    # parallel call to _calculate_permutation_scores, irrespective of
    # the fact that variables are shared or not depending on the active
    # joblib backend (sequential, thread-based or process-based).
    random_state = check_random_state(random_state)
    random_seed = random_state.randint(np.iinfo(np.int32).max + 1)

    scorer = check_scoring(estimator, scoring=scoring)
    baseline_score = scorer(estimator, X, y)

    scores = Parallel(n_jobs=n_jobs)(delayed(_calculate_permutation_scores)(
        estimator, X, y, col_idx, random_seed, n_repeats, scorer
    ) for col_idx in range(X.shape[1]))

    importances = baseline_score - np.array(scores)
    return Bunch(importances_mean=np.mean(importances, axis=1),
                 importances_std=np.std(importances, axis=1),
                 importances=importances)

if __name__=='__main__':

    time_point = 23  # this means 24 hours because of python's indexing
    batch_size = 2

    base_dir = 'models/experiments/final/TPC'
    exp_dir = 'models/experiments/final/TPC/2020-06-27_2048171'

    tpc = TPC(base_dir=base_dir,
              resume=exp_dir,
              resume_save_types=('model',
                                 'simple',
                                 'th_vars',
                                 'results'))

    c = initialise_tpc_arguments()
    c['mode'] = 'test'
    c['exp_name'] = 'TPC'
    c['model_type'] = 'tpc'
    tpc.config = c

    tpc.setup()
    tpc._setup_internal()
    tpc.prepare()
    tpc.model.eval()

    test_batches = tpc.test_datareader.batch_gen(batch_size=batch_size)
    ig = IntegratedGradients(tpc.model)

    N = 0
    target = time_point - 5  # The minus 5 is to compensate for the first five hours where predictions are not given.

    f1 = '-lymphs'
    f2 = 'potassium'
    f3 = 'glucose'
    f4 = 'Non-Invasive BP'
    f5 = 'heartrate'
    f6 = 'respiration'

    predictions = [0, 0, 0, 0, 0]
    importances = {}
    for f in (f1, f2, f3, f4, f5, f6):
        importances[f] = [0, 0, 0, 0, 0]
    feature_names = list(pd.read_csv('eICU_data/test/timeseries.csv', nrows=0).columns)
    feature_names.pop(0)

    for i, (padded, mask, diagnoses, flat, labels, seq_lengths) in enumerate(test_batches):
        if i == 11//batch_size:
            for l in range(int(seq_lengths[1].item()) - 5):
                print(l)

                attr = ig.attribute((padded, diagnoses, flat), target=l)
                predictions.append(tpc.model(padded, diagnoses, flat).item())

                # day_data is an array containing the indices of patients who stayed at least 24 hours (or up to `timepoint')
                day_data = mask[:, target].cpu().numpy().flatten().nonzero()[0]
                n = len(day_data)
                ts_attr = attr[0].detach()[:, :, :time_point].cpu().numpy()[day_data]
                abs_ts_attr = np.abs(ts_attr)
                ts_fts = padded[:, :, :time_point].cpu().numpy()[day_data]
                abs_ts_fts = np.abs(ts_fts)
                ts_nonzero = (abs_ts_fts != 0).sum(axis=2)
                # we only include patients who are indexed in `day_data' from now on, we can define the number of these patients as B
                # ts is an array containing the sum of the absolute values for the integrated gradient attributions (the sum is taken across timepoints)
                abs_ts_attr = abs_ts_attr.sum(axis=2) / ts_nonzero  # B x (2F + 2)

                for f in (f1, f2, f3, f4, f5, f6):
                    importances[f].append(abs_ts_attr[1][feature_names.index(f)])

            break



    mpl.rc('font', family = 'serif', size = 14)
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    plt.figure(figsize=(9,5))

    print(predictions)

    test = pd.read_csv('/Users/emmarocheteau/PycharmProjects/los-discharge-prediction/eICU_data/test/timeseries.csv', nrows=1000)
    test = test.loc[test['patient'] == 3132400]
    test_df = pd.DataFrame([test[f1].values, test[f1].values, importances[f1],
                            test[f2].values, test[f2].values, importances[f2],
                            test[f3].values, test[f3].values, importances[f3],
                            test[f4].values, test[f4].values, importances[f4],
                            test[f5].values, test[f5].values, importances[f5],
                            test[f6].values, test[f6].values, importances[f6]])
    plt.pcolor(test_df, cmap='coolwarm', vmax=1.2, vmin=-1.2)
    plt.ylim((0, 17))
    plt.yticks([1, 4, 7, 10, 13, 16], [f1, f2, f3, f4, f5, f6])
    plt.colorbar()
    plt.xlabel('Time since ICU admission (hours)')
    plt.savefig('figures/single_patient_importance.png', dpi=300, bbox_inches='tight')

