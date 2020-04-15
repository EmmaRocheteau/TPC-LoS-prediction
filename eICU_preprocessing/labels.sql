-- MUST BE RUN FIRST

-- creates a materialized view labels which looks like this:
/*
 uniquepid | patienthealthsystemstayid | patientunitstayid | unitvisitnumber | predictedhospitalmortality | actualhospitalmortality |  predictediculos  | actualiculos
-----------+---------------------------+-------------------+-----------------+----------------------------+-------------------------+-------------------+--------------
 002-34521 |                    128952 |            141208 |               1 | 2.2488507483439711E-3      | ALIVE                   | 0.444563334605296 |          0.5
 002-8979  |                    128973 |            141233 |               1 | 3.5490436381315972E-2      | ALIVE                   |   3.1075395670653 |      10.8923
 002-24408 |                    128999 |            141265 |               1 | 5.2291724973649173E-2      | ALIVE                   |  2.04551427140529 |       4.2138
 002-67735 |                    129020 |            141288 |               1 | 2.6198653111753761E-2      | ALIVE                   |  4.63527255617907 |       1.1326
 002-30269 |                    129026 |            141296 |               1 | 0.25711724536512859        | EXPIRED                 |  4.84374683684138 |       1.9638
*/

--hospitalid - which uniquely identifies each hospital in the database.
--uniquepid - uniquely identifies patients (i.e. it is always the same value for the same person)
--patienthealthsystemsstayid - uniquely identifies hospitals stays
--patientunitstayid - uniquely identifies unit stays (usually the unit is an ICU within a hospital)

-- delete the materialized view labels if it already exists
drop materialized view if exists ld_labels cascade;
create materialized view ld_labels as
  -- select all the data we need from the apache predictions table, plus patient identifier and hospital identifier
  select p.uniquepid, p.patienthealthsystemstayid, apr.patientunitstayid, p.unitvisitnumber, p.unitdischargelocation,
    p.unitdischargeoffset, p.unitdischargestatus, apr.predictedhospitalmortality, apr.actualhospitalmortality,
    apr.predictediculos, apr.actualiculos
    from patient as p
    inner join apachepatientresult as apr
      on p.patientunitstayid = apr.patientunitstayid
    where apr.apacheversion = 'IVa'  -- only use the most recent apache prediction model
      and apr.actualiculos > (5/24)  -- and exclude anyone who doesn't have at least 5 hours of data
      and nullif(replace(p.age, '> 89', '89'), '')::int > 17;  -- only include adults