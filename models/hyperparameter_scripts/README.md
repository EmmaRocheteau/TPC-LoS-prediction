Hyperparameter tuning
===============================

Please note that all these hyperparameter scripts optimise parameters on the length of stay (LoS) task only. This is 
because our paper is mainly concerned with optimising for LoS. We adopt these best hyperparameters for all the final 
experiments.

The hyperparameter tuning can be replicated by running commands such as:

```
python -m models.hyperparameter_scripts.eICU.tpc
```

Trixi provides a useful way to visualise effects of the hyperparameters (after running the following command, navigate to http://localhost:8080 in your browser):

```
python -m trixi.browser --port 8080 models/experiments/hyperparameters/eICU/TPC
```