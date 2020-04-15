eICU Preprocessing
==================================

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
5) To run the preprocessing scripts in your terminal:
    ```
    python3 {path_to_repository}/eICU_preprocessing/run_all_preprocessing.py
    ```
