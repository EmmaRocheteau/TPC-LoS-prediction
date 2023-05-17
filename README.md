[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporal-pointwise-convolutional-networks-for/predicting-patient-outcomes-on-eicu)](https://paperswithcode.com/sota/predicting-patient-outcomes-on-eicu?p=temporal-pointwise-convolutional-networks-for)

Patient Outcome Prediction with TPC Networks
===============================

This repository contains the code used for **Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit** (published at **ACM CHIL 2021**) and implementation instructions. You can watch a brief project talk here:

[![Watch the video](https://i.ytimg.com/vi/bDRbATjlUmY/maxresdefault.jpg)](https://www.youtube.com/watch?v=bDRbATjlUmY)
 
## Citation
If you use this code or the models in your research, please cite the following:

```
@inproceedings{rocheteau2021,
author = {Rocheteau, Emma and Li\`{o}, Pietro and Hyland, Stephanie},
title = {Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit},
year = {2021},
isbn = {9781450383592},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3450439.3451860},
doi = {10.1145/3450439.3451860},
booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
pages = {58–68},
numpages = {11},
keywords = {intensive care unit, length of stay, temporal convolution, mortality, patient outcome prediction},
location = {Virtual Event, USA},
series = {CHIL '21}
}
```

## Motivation
The pressure of ever-increasing patient demand and budget restrictions make hospital bed management a daily challenge for clinical staff. Most critical is the efficient allocation of resource-heavy Intensive Care Unit (ICU) beds to the patients who need life support. Central to solving this problem is knowing for how long the current set of ICU patients are likely to stay in the unit. In this work, we propose a new deep learning model based on the combination of temporal convolution and pointwise (1x1) convolution, to solve the length of stay prediction task on the eICU and MIMIC-IV critical care datasets. The model – which we refer to as Temporal Pointwise Convolution (TPC) – is specifically designed to mitigate common challenges with Electronic Health Records, such as skewness, irregular sampling and missing data. In doing so, we have achieved significant performance benefits of 18-68% (metric and dataset dependent) over the commonly used Long-Short Term Memory (LSTM) network, and the multi-head self-attention network known as the Transformer. By adding mortality prediction as a side-task, we can improve performance further still, resulting in a mean absolute deviation of 1.55 days (eICU) and 2.28 days (MIMIC-IV) on predicting remaining length of stay.

## Headline Results

### Length of Stay Prediction

We report on the following metrics: 
- Mean absolute deviation (MAD)
- Mean absolute percentage error (MAPE)
- Mean squared error (MSE)
- Mean squared log error (MSLE)
- Coefficient of determination (R<sup>2</sup>)
- Cohen Kappa Score (Harutyunyan et al. 2019)

For the first four metrics, lower is better. For the last two, higher is better.

#### eICU

Model | MAD | MAPE | MSE | MSLE | R<sup>2</sup> | Kappa
--- | --- | --- | --- | --- | --- | ---
Mean* | 3.21 | 395.7 | 29.5 | 2.87 | 0.00 | 0.00
Median* | 2.76 | 184.4 | 32.6 | 2.15 | -0.11 | 0.00
LSTM | 2.39±0.00 | 118.2±1.1 | 26.9±0.1 | 1.47±0.01 | 0.09±0.00 | 0.28±0.00
CW LSTM | 2.37±0.00 | 114.5±0.4 | 26.6±0.1 | 1.43±0.00 | 0.10±0.00 | 0.30±0.00
Transformer | 2.36±0.00 | 114.1±0.6 | 26.7±0.1 | 1.43±0.00 | 0.09±0.00 | 0.30±0.00
TPC | 1.78±0.02 | 63.5±4.3 | 21.7±0.5 | 0.70±0.03 | 0.27±0.02 | 0.58±0.01

Our model (TPC) significantly outperforms all baselines by large margins. 
*The mean and median "models" always predict 3.47 and 1.67 days respectively (the mean and median of the training set).

#### MIMIC-IV

Please note that this is not the same cohort as used in Harutyunyan et al. 2019. They use the older MIMIC-III database and I have developed my own preprocessing pipeline to closely match that of eICU.

Model | MAD | MAPE | MSE | MSLE | R<sup>2</sup> | Kappa
--- | --- | --- | --- | --- | --- | ---
Mean* | 5.24 | 474.9 | 77.7 | 2.80 | 0.000.00
Median* | 4.60 | 216.8 | 86.8 | 2.09 | -0.12 | 0.00
LSTM | 3.68±0.02 | 107.2±3.1 | 65.7±0.7 | 1.26±0.01 | 0.15±0.01 | 0.43±0.01
CW LSTM | 3.68±0.02 | 107.0±1.8 | 66.4±0.6 | 1.23±0.01 | 0.15±0.01 | 0.43±0.00
Transformer | 3.62±0.02 | 113.8±1.8 | 63.4±0.5 | 1.21±0.01 | 0.18±0.01 | 0.45±0.00
TPC | 2.39±0.03 | 47.6±1.4 | 46.3±1.3 | 0.39±0.02 | 0.40±0.02 | 0.78±0.01

*The mean and median "models" always predict 5.70 and 2.70 days respectively (the mean and median of the training set).

### Mortality Prediction

We report on the following metrics: 
- Area under the receiver operating characteristic curve (AUROC)
- Area under the precision recall curve (AUPRC)

For both metrics, higher is better.

#### eICU

Model | AUROC | AUPRC
--- | --- | --- 
LSTM | 0.849±0.002 | 0.407±0.012
CW LSTM | 0.855±0.001 | 0.464±0.004
Transformer | 0.851±0.002 | 0.454±0.005
TPC | 0.864±0.001 | 0.508±0.005

#### MIMIC-IV

Model | AUROC | AUPRC
--- | --- | --- 
LSTM | 0.895±0.001 | 0.657±0.003 
CW LSTM | 0.897±0.002 | 0.650±0.005
Transformer | 0.890±0.002 | 0.641±0.008
TPC | 0.905±0.001 | 0.691±0.006

### Multitask Prediction

These are the results when the model is trained to solve length of stay and mortality at the same time.

#### eICU

Model | AUROC | AUPRC | MAD | MAPE | MSE | MSLE | R<sup>2</sup> | Kappa
--- | --- | --- | --- | --- | --- | --- | --- | ---
LSTM | 0.852±0.003 | 0.436±0.007 | 2.40±0.01 | 116.5±0.8 | 27.2±0.2 | 1.47±0.01 | 0.08±0.01 | 0.28±0.01
CW LSTM | 0.865±0.002 | 0.490±0.007 | 2.37±0.00 | 115.0±0.7 | 26.8±0.1 | 1.44±0.00 | 0.09±0.00 | 0.30±0.00
Transformer | 0.858±0.001 | 0.475±0.004 | 2.36±0.00 | 114.2±0.7 | 26.6±0.1 | 1.43±0.00 | 0.10±0.00 | 0.30±0.00
TPC | 0.865±0.002 | 0.523±0.006 | 1.55±0.01 | 46.4±2.6 | 18.7±0.2 | 0.40±0.02 | 0.37±0.01 | 0.70±0.00

#### MIMIC-IV

Model | AUROC | AUPRC | MAD | MAPE | MSE | MSLE | R<sup>2</sup> | Kappa
--- | --- | --- | --- | --- | --- | --- | --- | ---
LSTM | 0.896±0.002 | 0.659±0.004 | 3.66±0.01 | 106.8±2.7 | 65.3±0.6 | 1.25±0.01 | 0.16±0.01 | 0.44±0.00
CW LSTM | 0.899±0.002 | 0.654±0.003 | 3.69±0.02 | 107.2±1.6 | 66.3±0.6 | 1.23±0.01 | 0.15±0.01 | 0.44±0.00
Transformer | 0.898±0.001 | 0.656±0.005 | 3.61±0.01 | 112.3±2.0 | 63.3±0.3 | 1.20±0.01 | 0.19±0.00 | 0.45±0.00
TPC | 0.918±0.002 | 0.713±0.007 | 2.28±0.07 | 32.4±1.2 | 42.0±1.2 | 0.19±0.00 | 0.46±0.02 | 0.85±0.00

## Pre-Processing Instructions

### eICU

1) To run the sql files you must have the eICU database set up: https://physionet.org/content/eicu-crd/2.0/. 

2) Follow the instructions: https://eicu-crd.mit.edu/tutorials/install_eicu_locally/ to ensure the correct connection configuration. 

3) Replace the eICU_path in `paths.json` to a convenient location in your computer, and do the same for `eICU_preprocessing/create_all_tables.sql` using find and replace for 
`'/Users/emmarocheteau/PycharmProjects/TPC-LoS-prediction/eICU_data/'`. Leave the extra '/' at the end.

4) In your terminal, navigate to the project directory, then type the following commands:

    ```
    psql 'dbname=eicu user=eicu options=--search_path=eicu'
    ```
    
    Inside the psql console:
    
    ```
    \i eICU_preprocessing/create_all_tables.sql
    ```
    
    This step might take a couple of hours.
    
    To quit the psql console:
    
    ```
    \q
    ```
    
5) Then run the pre-processing scripts in your terminal. This will need to run overnight:

    ```
    python3 -m eICU_preprocessing.run_all_preprocessing
    ```
    
### MIMIC-IV

1) To run the sql files you must have the MIMIC-IV database set up: https://physionet.org/content/mimiciv/0.4/. 

2) The official recommended way to access MIMIC-IV is via BigQuery: https://mimic-iv.mit.edu/docs/access/bigquery/. Personally I did not find it easy to store the necessary views and there is a 1GB size limit on the data you can download in the free tier, which is less than I am using here (the largest file to extract is timeseries.csv which is 4.49GB). However if you do wish to use BigQuery, note that you will have to make minor modifications to the code e.g. you would need to replace a reference to the table `patients` with `physionet-data.mimic_core.patients`. 
    
    Alternatively, you can follow instructions to set up the full database. The instructions for the previous version of MIMIC - MIMIC-III are here: https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/ for unix systems or: https://mimic.physionet.org/tutorials/install-mimic-locally-windows/ for windows. You will need to change `mimiciii` schema to `mimiciv` and use the files in: https://github.com/EmmaRocheteau/MIMIC-IV-Postgres in place of the files in: https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic/postgres (referenced in the instructions). Additionally you may find this resource helpful: https://github.com/MIT-LCP/mimic-iv/tree/master/buildmimic/postgres which is still in the process of being updated (as of November 2020).

3) Once you have a database connection, replace the MIMIC_path in `paths.json` to a convenient location in your computer, and do the same for `MIMIC_preprocessing/create_all_tables.sql` using find and replace for 
`'/Users/emmarocheteau/PycharmProjects/TPC-LoS-prediction/MIMIC_data/'`. Leave the extra '/' at the end.

4) If you have set up the database on your local computer, you can navigate to the project directory in your terminal, then type the following commands:

    ```
    psql 'dbname=mimic user=mimicuser options=--search_path=mimiciv'
    ```
    
    Inside the psql console:
    
    ```
    \i MIMIC_preprocessing/create_all_tables.sql
    ```
    
    This step might take a couple of hours.
    
    To quit the psql console:
    
    ```
    \q
    ```
    
5) Then run the pre-processing scripts in your terminal. This will need to run overnight:

    ```
    python3 -m MIMIC_preprocessing.run_all_preprocessing
    ```
    
   
## Running the models
1) Once you have run the pre-processing steps you can run all the models in your terminal. Set the working directory to the TPC-LoS-prediction, and run the following:

    ```
    python3 -m models.run_tpc
    ```
    
    Note that your experiment can be customised by using command line arguments e.g.
    
    ```
    python3 -m models.run_tpc --dataset eICU --task LoS --model_type tpc --n_layers 4 --kernel_size 3 --no_temp_kernels 10 --point_size 10 --last_linear_size 20 --diagnosis_size 20 --batch_size 64 --learning_rate 0.001 --main_dropout_rate 0.3 --temp_dropout_rate 0.1 
    ```
    
    Each experiment you run will create a directory within models/experiments. The naming of the directory is based on 
    the date and time that you ran the experiment (to ensure that there are no name clashes). The experiments are saved 
    in the standard trixi format: https://trixi.readthedocs.io/en/latest/_api/trixi.experiment.html.
    
2) The hyperparameter searches can be replicated by running:

    ```
    python3 -m models.hyperparameter_scripts.eICU.tpc
    ```
 
    Trixi provides a useful way to visualise effects of the hyperparameters (after running the following command, navigate to http://localhost:8080 in your browser):
    
    ```
    python3 -m trixi.browser --port 8080 models/experiments/hyperparameters/eICU/TPC
    ```
    
    The final experiments for the paper are found in models/final_experiment_scripts e.g.:
    
    ```
    python3 -m models.final_experiment_scripts.eICU.LoS.tpc
    ```
    
## References
Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, Greg Ver Steeg, and Aram Galstyan. Multitask Learning and Benchmarking with Clinical Time Series Data. Scientific Data, 6(96), 2019.
