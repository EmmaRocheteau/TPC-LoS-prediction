MIMIC Preprocessing
==================================

1) To run the sql files you must have the MIMIC-IV database set up: https://physionet.org/content/mimiciv/0.4/. 

2) The official recommended way to access MIMIC-IV is via BigQuery: https://mimic-iv.mit.edu/docs/access/bigquery/. Personally I did not find it easy to store the necessary views and there is a size limit on the data you can save in the free tier (1GB) which is less than I am using here. However if you do wish to use BigQuery, note that you will have to make minor modifications to the code e.g. you would need to replace a reference to the table `patients` with `physionet-data.mimic_core.patients`. 
    
    Alternatively, you can follow instructions to set up the full database. The instructions for the previous version of MIMIC - MIMIC-III are here: https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/ for unix systems or: https://mimic.physionet.org/tutorials/install-mimic-locally-windows/ for windows. You will need to change `mimiciii` schema to `mimiciv` and use the files in: https://github.com/EmmaRocheteau/MIMIC-IV-Postgres in place of the files in: https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic/postgres (referenced in the instructions).

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
    
    To quit the psql console:
    
    ```
    \q
    ```
    
    This step might take a couple of hours.
    
5) Then run the pre-processing scripts in your terminal. This will need to run overnight:

    ```
    python MIMIC_preprocessing/run_all_preprocessing.py
    ```