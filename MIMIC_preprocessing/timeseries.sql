-- extract the most common lab tests and the corresponding counts of how many patients have values for those labs
drop materialized view if exists ld_commonlabs cascade;
create materialized view ld_commonlabs as
  -- extracting the itemids for all the labevents that occur within the time bounds for our cohort
  with labsstay as (
    select l.itemid, la.stay_id
    from labevents as l
    inner join ld_labels as la
      on la.hadm_id = l.hadm_id
    where l.valuenum is not null  -- stick to the numerical data
      -- epoch extracts the number of seconds since 1970-01-01 00:00:00-00, we want to extract measurements between
      -- admission and the end of the patients' stay
      and (date_part('epoch', l.charttime) - date_part('epoch', la.intime))/(60*60*24) between -1 and la.los),
  -- getting the average number of times each itemid appears in an icustay (filtering only those that are more than 2)
  avg_obs_per_stay as (
    select itemid, avg(count) as avg_obs
    from (select itemid, count(*) from labsstay group by itemid, stay_id) as obs_per_stay
    group by itemid
    having avg(count) > 3)  -- we want the features to have at least 3 values entered for the average patient
  select d.label, count(distinct labsstay.stay_id) as count, a.avg_obs
    from labsstay
    inner join d_labitems as d
      on d.itemid = labsstay.itemid
    inner join avg_obs_per_stay as a
      on a.itemid = labsstay.itemid
    group by d.label, a.avg_obs
    -- only keep data that is present at some point for at least 25% of the patients, this gives us 45 lab features
    having count(distinct labsstay.stay_id) > (select count(distinct stay_id) from ld_labels)*0.25
    order by count desc;

-- get the time series features from the most common lab tests (45 of these)
drop materialized view if exists ld_timeserieslab cascade;
create materialized view ld_timeserieslab as
  -- we extract the number of minutes in labresultoffset because this is how the data in eICU is arranged
  select la.stay_id as patientunitstayid, floor((date_part('epoch', l.charttime) - date_part('epoch', la.intime))/60)
  as labresultoffset, d.label as labname, l.valuenum as labresult
    from labevents as l
    inner join d_labitems as d
      on d.itemid = l.itemid
    inner join ld_commonlabs as cl
      on cl.label = d.label  -- only include the common labs
    inner join ld_labels as la
      on la.hadm_id = l.hadm_id  -- only extract data for the cohort
    -- epoch extracts the number of seconds since 1970-01-01 00:00:00-00, we want to extract measurements between
    -- admission and the end of the patients' stay
    where (date_part('epoch', l.charttime) - date_part('epoch', la.intime))/(60*60*24) between -1 and la.los
      and l.valuenum is not null;  -- filter out null values

-- extract the most common chartevents and the corresponding counts of how many patients have values for those chartevents
drop materialized view if exists ld_commonchart cascade;
create materialized view ld_commonchart as
  -- extracting the itemids for all the chartevents that occur within the time bounds for our cohort
  with chartstay as (
      select ch.itemid, la.stay_id
        from chartevents as ch
        inner join ld_labels as la
          on la.stay_id = ch.stay_id
        where ch.valuenum is not null  -- stick to the numerical data
          -- epoch extracts the number of seconds since 1970-01-01 00:00:00-00, we want to extract measurements between
          -- admission and the end of the patients' stay
          and (date_part('epoch', ch.charttime) - date_part('epoch', la.intime))/(60*60*24) between -1 and la.los),
  -- getting the average number of times each itemid appears in an icustay (filtering only those that are more than 5)
  avg_obs_per_stay as (
    select itemid, avg(count) as avg_obs
    from (select itemid, count(*) from chartstay group by itemid, stay_id) as obs_per_stay
    group by itemid
    having avg(count) > 5)  -- we want the features to have at least 5 values entered for the average patient
  select d.label, count(distinct chartstay.stay_id) as count, a.avg_obs
    from chartstay
    inner join d_items as d
      on d.itemid = chartstay.itemid
    inner join avg_obs_per_stay as a
      on a.itemid = chartstay.itemid
    group by d.label, a.avg_obs
    -- only keep data that is present at some point for at least 25% of the patients, this gives us 129 chartevents features
    having count(distinct chartstay.stay_id) > (select count(distinct stay_id) from ld_labels)*0.25
    order by count desc;

-- get the time series features from the most common chart features (129 of these)
drop materialized view if exists ld_timeseries cascade;
create materialized view ld_timeseries as
  -- we extract the number of minutes in chartoffset because this is how the data in eICU is arranged
  select la.stay_id as patientunitstayid, floor((date_part('epoch', ch.charttime) - date_part('epoch', la.intime))/60)
  as chartoffset, d.label as chartvaluelabel, ch.valuenum as chartvalue
    from chartevents as ch
    inner join d_items as d
      on d.itemid = ch.itemid
    inner join ld_commonchart as cch
      on cch.label = d.label  -- only include the common chart features
    inner join ld_labels as la
      on la.stay_id = ch.stay_id  -- only extract data for the cohort
    where (date_part('epoch', ch.charttime) - date_part('epoch', la.intime))/(60*60*24) between -1 and la.los
      and ch.valuenum is not null;  -- filter out null values