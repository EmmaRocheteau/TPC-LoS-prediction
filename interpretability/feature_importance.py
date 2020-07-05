from models.run_tpc import TPC
import numpy as np
from models.initialise_arguments import initialise_tpc_arguments
from captum.attr import IntegratedGradients
from torch import cat, ones

time_point = 23  # this means 24 hours because of python's indexing
batch_size = 4

if __name__=='__main__':

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
    test_loss = []
    test_y_hat = np.array([])
    test_y = np.array([])

    test_attr_ts = np.zeros((176, time_point))
    test_attr_diag = np.zeros(293)
    test_attr_flat = np.zeros(65)
    N = 0

    for i, (padded, mask, diagnoses, flat, labels, seq_lengths) in enumerate(test_batches):

        target = time_point - 5

        # if at least one sequence is longer than 24hrs
        if max(seq_lengths.cpu().numpy()) - target > 0:

            # The minus 5 is to compensate for the first five hours where predictions are not given.
            attr, delta = ig.attribute((padded, diagnoses, flat), target=target, return_convergence_delta=True)

            # delete data that was less than a day
            day_data = mask[:, target].unsqueeze(1)
            ts = attr[0].detach()[:, :, :time_point] * day_data.unsqueeze(1)
            diag = attr[1].detach() * day_data
            flat = attr[2].detach() * day_data

            # keep a running sum of the feature attributions
            test_attr_ts = np.add(test_attr_ts, ts.cpu().numpy().sum(axis=0))
            test_attr_diag = np.add(test_attr_diag, diag.cpu().numpy().sum(axis=0))
            test_attr_flat = np.add(test_attr_flat, flat.cpu().numpy().sum(axis=0))

            N += int(sum(day_data))

            if i % 10 == 0:
                print('Done ' + str(N + 1))
                np.savetxt('interpretability/ts.csv', test_attr_ts/N, delimiter=',', fmt='%f')
                np.savetxt('interpretability/diag.csv', test_attr_diag/N, delimiter=',', fmt='%f')
                np.savetxt('interpretability/flat.csv', test_attr_flat/N, delimiter=',', fmt='%f')
