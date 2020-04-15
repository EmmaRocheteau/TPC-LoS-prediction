Running the models
==================================

1) Follow the instructions in the `README.md` file in the eICU_preprocessing directory.
2) You can run the tpc and lstm models with:
    ```
    python3 -m models.run_tpc
    python3 -m models.run_lstm
    ```
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

Running the models
==================================
