from models.run_tpc import TPC
import numpy as np
from models.initialise_arguments import initialise_tpc_arguments
from captum.attr import IntegratedGradients
from pathlib import Path

time_point = 23  # this means 24 hours because of python's indexing
batch_size = 2

p = Path('interpretability').glob('**/*.csv')
csv_files = [x for x in p if x.is_file()]
for c in csv_files:
    c.unlink()

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

ig = IntegratedGradients(tpc.model)

test_batches = tpc.test_datareader.batch_gen(batch_size=batch_size)

N = 0
target = time_point - 5  # The minus 5 is to compensate for the first five hours where predictions are not given.

for i, (padded, mask, diagnoses, flat, labels, seq_lengths) in enumerate(test_batches):

    # if at least one sequence is longer than 24hrs
    if max(seq_lengths.cpu().numpy()) - target > 0:

        attr = ig.attribute((padded, diagnoses, flat), target=target)

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
        abs_ts_attr = abs_ts_attr.sum(axis=2)/ts_nonzero  # B x (2F + 2)
        ts_attr = ts_attr.reshape(n * 23, -1)
        ts_attr[ts_attr == 0] = 'nan'
        diag_attr = attr[1].detach().cpu().numpy()[day_data]  # B x D
        diag_attr[diag_attr == 0] = 'nan'
        flat_attr = attr[2].detach().cpu().numpy()[day_data]  # B x f
        flat_attr[flat_attr == 0] = 'nan'

        abs_ts_fts = abs_ts_fts.sum(axis=2)/ts_nonzero  # B x (2F + 2)
        ts_fts = ts_fts.reshape(n * 23, -1)
        ts_fts[ts_fts == 0] = 'nan'
        diag_fts = diagnoses.cpu().numpy()[day_data]  # B x D
        diag_fts[diag_fts == 0] = 'nan'
        flat_fts = flat.cpu().numpy()[day_data]  # B x f
        flat_fts[flat_fts == 0] = 'nan'

        for ar, fname in ((abs_ts_attr, 'abs_ts_attr24.csv'),
                          (flat_attr, 'flat_attr24.csv'),
                          (diag_attr, 'diag_attr.csv'),
                          (abs_ts_fts, 'abs_ts_fts24.csv'),
                          (flat_fts, 'flat_fts24.csv'),
                          (diag_fts, 'diag_fts.csv')):
            with open('interpretability/'+fname, 'ba') as f:
                np.savetxt(f, ar, delimiter=',', fmt='%f')

        N += n

    if i % 100 == 0:
        print('Done ' + str(N))