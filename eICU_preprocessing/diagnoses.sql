-- MUST BE RUN AFTER labels.sql

-- creates a materialized view diagnoses which looks like this:
/*
 patientunitstayid |                                                           diagnosisstring
-------------------+--------------------------------------------------------------------------------------------------------------------------------------
            141168 | cardiovascular|arrhythmias|atrial fibrillation|with hemodynamic compromise
            141168 | cardiovascular|chest pain / ASHD|coronary artery disease|known
            141168 | cardiovascular|ventricular disorders|cardiomyopathy
            141168 | cardiovascular|ventricular disorders|congestive heart failure
            141168 | notes/Progress Notes/Past History/Organ Systems/Cardiovascular (R)/AICD/AICD
            141168 | notes/Progress Notes/Past History/Organ Systems/Cardiovascular (R)/Arrhythmias/atrial fibrillation - chronic
            141168 | notes/Progress Notes/Past History/Organ Systems/Cardiovascular (R)/Congestive Heart Failure/CHF - class II
            141168 | notes/Progress Notes/Past History/Organ Systems/Cardiovascular (R)/Congestive Heart Failure/CHF - severity unknown
*/

-- delete the materialized view diagnoses if it already exists
drop materialized view if exists ld_diagnoses cascade;
create materialized view ld_diagnoses as
  -- for current diagnoses:
  select d.patientunitstayid, d.diagnosisstring
    from diagnosis as d
    -- restrict only to the patients present in the labels materialized view
    inner join ld_labels as l on l.patientunitstayid = d.patientunitstayid
    inner join patient as p on p.patientunitstayid = d.patientunitstayid
    -- make sure the diagnosis was entered either before the ICU admission, or within the first 5 hours
    where d.diagnosisoffset < 60*5 -- corresponds to 5 hours into the stay
  union
  -- for past medical history:
  select ph.patientunitstayid, ph.pasthistorypath as diagnosisstring
    from pasthistory as ph
    inner join ld_labels as l on l.patientunitstayid = ph.patientunitstayid
    inner join patient as p on p.patientunitstayid = ph.patientunitstayid
    where ph.pasthistoryoffset < 60*5  -- corresponds to 5 hours into the stay
  union
  -- for admission diagnoses:
  select ad.patientunitstayid, ad.admitdxpath as diagnosisstring
    from admissiondx as ad
    inner join ld_labels as l on l.patientunitstayid = ad.patientunitstayid
    inner join patient as p on p.patientunitstayid = ad.patientunitstayid
    where ad.admitdxenteredoffset < 60*5;  -- corresponds to 5 hours into the stay