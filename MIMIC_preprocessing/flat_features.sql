/* Useful explorative query for finding variables to include in chartevents
select count(distinct ch.stay_id), d.label, avg(valuenum) as avg_value
  from chartevents as ch
    inner join icustays as i
      on ch.stay_id = i.stay_id
    inner join d_items as d
      on d.itemid = ch.itemid
    where lower(d.label) like '%height%' -- <-- enter phrase of interest
      and date_part('hour', ch.charttime) - date_part('hour', i.intime) < 5
    group by d.label;
*/

-- requires tablefunc extension, can be obtained with 'CREATE EXTENSION tablefunc;'
drop materialized view if exists extra_vars cascade;
create materialized view extra_vars as
  select * from crosstab(
    'select ch.stay_id, d.label, avg(valuenum) as value
      from chartevents as ch
        inner join icustays as i
          on ch.stay_id = i.stay_id
        inner join d_items as d
          on d.itemid = ch.itemid
        where ch.valuenum is not null
          and d.label in (''Admission Weight (Kg)'', ''GCS - Eye Opening'', ''GCS - Motor Response'', ''GCS - Verbal Response'', ''Height (cm)'')
          and ch.valuenum != 0
          and date_part(''hour'', ch.charttime) - date_part(''hour'', i.intime) between -24 and 5
        group by ch.stay_id, d.label'
        ) as ct(stay_id integer, weight double precision, eyes double precision, motor double precision, verbal double precision, height double precision);


drop materialized view if exists ld_flat cascade;
create materialized view ld_flat as
  select distinct i.stay_id as patientunitstayid, p.gender, (extract(year from i.intime) - p.anchor_year + p.anchor_age) as age,
    adm.ethnicity, i.first_careunit, adm.admission_location, adm.insurance, ev.height, ev.weight,
    extract(hour from i.intime) as hour, ev.eyes, ev.motor, ev.verbal
    from ld_labels as la
    inner join patients as p on p.subject_id = la.subject_id
    inner join icustays as i on i.stay_id = la.stay_id
    inner join admissions as adm on adm.hadm_id = la.hadm_id
    left join extra_vars as ev on ev.stay_id = la.stay_id;