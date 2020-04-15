-- MUST BE RUN AFTER labels.sql

-- creates a materialized view flat which looks like this:
/*
 patientunitstayid | gender | age | ethnicity | admissionheight | admissionweight |               apacheadmissiondx               | hour |   unittype   | unitadmitsource | unitvisitnumber | unitstaytype |     physicianspeciality      | intubated | vent | dialysis | eyes | motor | verbal | meds | bedcount | numbedscategory | teachingstatus | region
-------------------+--------+-----+-----------+-----------------+-----------------+-----------------------------------------------+------+--------------+-----------------+-----------------+--------------+------------------------------+-----------+------+----------+------+-------+--------+------+----------+-----------------+----------------+---------
            141168 | Female | 70  | Caucasian |          152.40 |           84.30 | Rhythm disturbance (atrial, supraventricular) |   15 | Med-Surg ICU | Direct Admit    |               1 | admit        | critical care medicine (CCM) |         0 |    0 |        0 |    4 |     6 |      5 |    0 |       12 | <100            | f              | Midwest
            141194 | Male   | 68  | Caucasian |          180.30 |           73.90 | Sepsis, renal/UTI (including bladder)         |    7 | CTICU        | Floor           |               1 | admit        | critical care medicine (CCM) |         0 |    0 |        0 |    3 |     6 |      4 |    0 |       38 | >= 500          | t              | Midwest
            141203 | Female | 77  | Caucasian |          160.00 |           70.20 | Arrest, respiratory (without cardiac arrest)  |   20 | Med-Surg ICU | Floor           |               1 | admit        | hospitalist                  |         0 |    1 |        0 |    1 |     3 |      1 |    0 |       18 | 100 - 249       | f              | Midwest
            141227 | Male   | 82  | Caucasian |          185.40 |           82.20 | Sepsis, pulmonary                             |   12 | Med-Surg ICU | Floor           |               1 | admit        | internal medicine            |         0 |    1 |        0 |    3 |     6 |      4 |    0 |        9 | <100            | f              | Midwest
            141233 | Female | 81  | Caucasian |          165.10 |           61.70 | Mitral valve replacement                      |   17 | CTICU        | Operating Room  |               1 | admit        | surgery-cardiac              |         1 |    1 |        0 |    4 |     6 |      5 |    0 |       38 | >= 500          | t              | Midwest
            141244 | Male   | 59  | Caucasian |          180.30 |           92.30 | Graft, femoral-popliteal bypass               |    2 | CTICU        | Operating Room  |               1 | admit        | critical care medicine (CCM) |         0 |    0 |        0 |    4 |     6 |      5 |    0 |       38 | >= 500          | t              | Midwest
*/

-- delete the materialized view flat if it already exists
drop materialized view if exists ld_flat cascade;
create materialized view ld_flat as
  -- for some reason lots of multiple records are produced, the distinct gets rid of these
  select distinct la.patientunitstayid, p.gender, p.age, p.ethnicity, p.admissionheight, p.admissionweight,
    p.apacheadmissiondx, extract(hour from to_timestamp(p.unitadmittime24,'HH24:MI:SS')) as hour, p.unittype,
    p.unitadmitsource, p.unitvisitnumber, p.unitstaytype, apr.physicianspeciality, aps.intubated, aps.vent,
    aps.dialysis, aps.eyes, aps.motor, aps.verbal, aps.meds, apv.bedcount, h.numbedscategory, h.teachingstatus,
    h.region
    from patient as p
    inner join apacheapsvar as aps on aps.patientunitstayid = p.patientunitstayid
    inner join apachepatientresult as apr on apr.patientunitstayid = p.patientunitstayid
    inner join apachepredvar as apv on apv.patientunitstayid = p.patientunitstayid
    inner join hospital as h on h.hospitalid = p.hospitalid
    inner join ld_labels as la on la.patientunitstayid = p.patientunitstayid;
