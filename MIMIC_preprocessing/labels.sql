-- delete the materialized view labels if it already exists
drop materialized view if exists ld_labels cascade;
create materialized view ld_labels as
  select i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, adm.hospital_expire_flag, i.los
    from icustays as i
    inner join admissions as adm
      on adm.hadm_id = i.hadm_id
    inner join patients as p
      on p.subject_id = i.subject_id
    where i.los > (5/24)  -- and exclude anyone who doesn't have at least 5 hours of data
      and (extract(year from i.intime) - p.anchor_year + p.anchor_age) > 17;  -- only include adults