# eICU Length of Stay Prediction

This repository contains the code used for Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit.

## Citation
If you use this code or the models in your research, please cite the following publication:

## Motivation
The pressure of ever-increasing patient demand and budget restrictions make hospital bed management a daily challenge 
for clinical staff. Most critical, is the efficient allocation of resource-heavy Intensive Care Unit (ICU) beds to the 
patients who need life support. Central to solving this problem is knowing for how long the current set of ICU patients 
are likely to stay in the unit. In this work we propose a new deep learning model based on the combination of temporal 
convolution and pointwise (or 1x1) convolution, to solve the length of stay prediction task on the eICU critical care 
dataset. The model — which we refer to as Temporal Pointwise Convolution (TPC) — was developed using a tailored, 
domain-specific approach. We specifically design the model to mitigate for common challenges with Electronic Health 
Records, such as skewness, irregular sampling and missing data. In doing so, we have achieved significant performance 
benefits of 22-59% (metric dependent) over the commonly used Long-Short Term Memory (LSTM) network.

## Running the models

### eICU Pre-processing


1) To run the sql files you must have the eICU database set up: https://physionet.org/content/eicu-crd/2.0/. 

2) Follow the instructions: https://eicu-crd.mit.edu/tutorials/install_eicu_locally/ to ensure the correct connection configuration. 

3) Replace the eICU_paths in `create_all_tables.sql` and `run_all_preprocessing.py` using find and replace for 
`'/Users/emmarocheteau/PycharmProjects/eICU-LoS-prediction/eICU_data/'` so that they run on your local computer.

4) In your terminal:

    ```
    psql 'dbname=eicu user=eicu options=--search_path=eicu'
    ```
    
    Inside the psql console:
    
    ```
    \i {path_to_repository}/eICU_preprocessing/create_all_tables.sql
    ```
    
    To quit the psql console:
    
    ```
    \q
    ```
    
5) Then run the pre-processing scripts in your terminal. This will need to run overnight:

    ```
    python3 {path_to_repository}/eICU_preprocessing/run_all_preprocessing.py

   
### Running the models


1) Once you have run the pre-processing steps you can run all the models in your terminal. Set the working directory to the eICU-LoS-prediction, and run the following:

    ```
    python3 -m models.run_tpc
    ```
    
    Note that your experiment can be customised by using command line arguments e.g.
    
    ```
    python3 -m models.run_tpc --model_type tpc --n_layers 4 --kernel_size 3 --no_temp_kernels 10 --point_size 10 --last_linear_size 20 --diagnosis_size 20 --batch_size 64 --learning_rate 0.001 --main_dropout_rate 0.3 --temp_dropout_rate 0.1 
    ```
    
    Each experiment you run will create a directory within models/experiments. The naming of the directory is based on 
    the date and time that you ran the experiment (to ensure that there are no name clashes). The experiments are saved 
    in the standard trixi format: https://trixi.readthedocs.io/en/latest/_api/trixi.experiment.html.
    
2) The hyperparameter searches can be replicated by running:

    ```
    python3 -m models.hyperparameters.tpc
    ```
 
    Trixi provides a useful way to visualise effects of the hyperparameters (after running the following command, navigate to http://localhost:8080 in your browser):
    
    ```
    python3 -m trixi.browser --port 8080 models/hyperparameters/TPC
    ```
    
    The final experiments for the paper are found in models/scripts:
    
    ```
    python3 -m models.scripts.tpc
    ```
