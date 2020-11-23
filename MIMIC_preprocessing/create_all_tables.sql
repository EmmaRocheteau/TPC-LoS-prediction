-- creates all the tables and produces csv files
-- takes a while to run (about an hour)

-- change the paths to those in your local computer using find and replace for '/Users/emmarocheteau/PycharmProjects/TPC-LoS-prediction/MIMIC_data/'.
-- keep the file names the same

\i MIMIC_preprocessing/labels.sql
\i MIMIC_preprocessing/flat_features.sql
\i MIMIC_preprocessing/timeseries.sql

-- we need to make sure that we have at least some form of time series for every patient in diagnoses, flat and labels
drop materialized view if exists ld_timeseries_patients cascade;
create materialized view ld_timeseries_patients as
  with repeats as (
    select distinct patientunitstayid
      from ld_timeserieslab
    union
    select distinct patientunitstayid
      from ld_timeseries)
  select distinct patientunitstayid
    from repeats;

-- renaming some of the variables so that they are equivalent to those in eICU
\copy (select subject_id as uniquepid, hadm_id as patienthealthsystemstayid, stay_id as patientunitstayid, hospital_expire_flag as actualhospitalmortality, los as actualiculos from ld_labels as l where l.stay_id in (select * from ld_timeseries_patients) order by l.stay_id) to '/Users/emmarocheteau/PycharmProjects/TPC-LoS-prediction/MIMIC_data/labels.csv' with csv header
\copy (select * from ld_flat as f where f.patientunitstayid in (select * from ld_timeseries_patients) order by f.patientunitstayid) to '/Users/emmarocheteau/PycharmProjects/TPC-LoS-prediction/MIMIC_data/flat_features.csv' with csv header
\copy (select * from ld_timeserieslab as tl order by tl.patientunitstayid, tl.labresultoffset) to '/Users/emmarocheteau/PycharmProjects/TPC-LoS-prediction/MIMIC_data/timeserieslab.csv' with csv header
\copy (select * from ld_timeseries as t order by t.patientunitstayid, t.chartoffset) to '/Users/emmarocheteau/PycharmProjects/TPC-LoS-prediction/MIMIC_data/timeseries.csv' with csv header