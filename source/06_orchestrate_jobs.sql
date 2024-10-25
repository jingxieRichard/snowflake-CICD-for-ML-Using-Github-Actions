-- SNOWFLAKE ADVANTAGE: Tasks (with Stream triggers)

USE ROLE ACCOUNTADMIN;
USE WAREHOUSE JING_TEST_WH;
USE DATABASE ML_SNOWPARK_CI_CD_DB;

-- ----------------------------------------------------------------------------
-- Step #1: Create Streams on Data
-- ----------------------------------------------------------------------------

CREATE OR REPLACE STREAM DATA_PROCESSING.APPLICATION_RECORD_STREAM ON TABLE DATA_PROCESSING.APPLICATION_RECORD;
CREATE OR REPLACE STREAM ML_PROCESSING.PROCESSED_INPUT_STREAM ON TABLE ML_PROCESSING.PROCESSED_INPUT;

-- ----------------------------------------------------------------------------
-- Step #2: Create the task to load data daily
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TASK ML_PROCESSING.TASK_PROCESS_INPUT;
WAREHOUSE = JING_TEST_WH
SCHEDULE = '720 MINUTE'
WHEN  
    SYSTEM$STREAM_HAS_DATA('ML_SNOWPARK_CI_CD_DB.DATA_PROCESSING.APPLICATION_RECORD_STREAM')
AS 
CALL ML_SNOWPARK_CI_CD_DB.ML_PROCESSING.SPROC_PROCESS_INPUT();

-- ----------------------------------------------------------------------------
-- Step #3: Create the task to do model inference
-- ----------------------------------------------------------------------------

-- Create task and schedule it to run on the stream every 12 hours 
CREATE OR REPLACE TASK ML_PROCESSING.TASK_MODEL_INFERENCE
WAREHOUSE = JING_TEST_WH
AFTER ML_PROCESSING.TASK_PROCESS_INPUT
WHEN SYSTEM$STREAM_HAS_DATA('ML_SNOWPARK_CI_CD_DB.ML_PROCESSING.PROCESSED_INPUT_STREAM')
AS 
INSERT INTO ML_SNOWPARK_CI_CD_DB.ML_PROCESSING.SCORED_DATA
(
    SELECT t.* exclude (METADATA$ACTION, METADATA$ISUPDATE, METADATA$ROW_ID, TIMESTAMP),
    ML_PROCESSING.PREDICT_DEFAULT(
        CODE_GENDER_F,
CODE_GENDER_M,
NAME_HOUSING_TYPE_COOPAPARTMENT,
NAME_HOUSING_TYPE_HOUSEAPARTMENT,
NAME_HOUSING_TYPE_MUNICIPALAPARTMENT,
NAME_HOUSING_TYPE_OFFICEAPARTMENT,
NAME_HOUSING_TYPE_RENTEDAPARTMENT,
NAME_HOUSING_TYPE_WITHPARENTS,
OCCUPATION_TYPE_ACCOUNTANTS,
OCCUPATION_TYPE_CLEANINGSTAFF,
OCCUPATION_TYPE_COOKINGSTAFF,
OCCUPATION_TYPE_CORESTAFF,
OCCUPATION_TYPE_DRIVERS,
OCCUPATION_TYPE_HRSTAFF,
OCCUPATION_TYPE_HIGHSKILLTECHSTAFF,
OCCUPATION_TYPE_ITSTAFF,
OCCUPATION_TYPE_LABORERS,
OCCUPATION_TYPE_LOWSKILLLABORERS,
OCCUPATION_TYPE_MANAGERS,
OCCUPATION_TYPE_MEDICINESTAFF,
OCCUPATION_TYPE_PRIVATESERVICESTAFF,
OCCUPATION_TYPE_REALTYAGENTS,
OCCUPATION_TYPE_SALESSTAFF,
OCCUPATION_TYPE_SECRETARIES,
OCCUPATION_TYPE_SECURITYSTAFF,
OCCUPATION_TYPE_WAITERSBARMENSTAFF,
OCCUPATION_TYPE_NONE,
AMT_INCOME_TOTAL,
DAYS_EMPLOYED,
FLAG_MOBIL,
CNT_FAM_MEMBERS
    ) AS PREDICTION,
CURRENT_TIMESTAMP AS PREDICTION_TIMESTAMP 
FROM ML_SNOWPARK_CI_CD_DB.ML_PROCESSING.PROCESSED_INPUT_STREAM as t
);

-- Step #4: Resume task to now run every 12 hours 
ALTER TASK ML_PROCESSING.TASK_MODEL_INFERENCE RESUME;
ALTER TASK ML_PROCESSING.TASK_PROCESS_INPUT RESUME;