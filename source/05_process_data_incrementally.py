import configparser
import sys
import os 
import json 
from datetime import datetime
import re 

from snowflake.snowpark.session import Session 
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T

from snowflake.ml.modeling.preprocessing import OneHotEncoder 

connection_params = {
    "connection_name": "default"
}

# To establish a connection to Snowflake, create a Snowpark session and pass connection_params as an argument
session = Session.builder.configs(connection_params).create()

session.use_database("ML_SNOWPARK_CI_CD_DB")
session.use_schema("ML_PROCESSING")

# Stored Proc to Process Data Incrementally 
def sproc_process_input(session: Session) -> T.Variant: 
    # Import Libraries 
    from datetime import datetime
    import re 
    import snowflake.snowpark.functions as F 
    from snowflake.ml.modeling.preprocessing import OneHotEncoder 

    # Create a Snowpark DataFrame 
    application_record_sdf = session.table("ML_SNOWPARK_CI_CD_DB.DATA_PROCESSING.APPLICATION_RECORD_STREAM")
    print("Application table size \t: ", application_record_sdf.count())

    if application_record_sdf.count() == 0:
        print("\n APPLICATION_RECORD_STREAM is empty")
        sys.exit()

    # Selecting a few columns for modeling 
    cols_numerical = ['AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 'CNT_FAM_MEMBERS']
    cols_categorical = ['CODE_GENDER', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
    application_record_sdf = application_record_sdf[cols_numerical+cols_categorical]

    # Perform one-hot-encoding for categorical columns 
    my_ohe_encoder = OneHotEncoder(input_cols=cols_categorical, output_cols=cols_categorical, drop_input_cols=True)
    prepared_sdf = my_ohe_encoder.fit(application_record_sdf).transform(application_record_sdf)

    # Clean column names to make it easier for future referencing 
    cols = prepared_sdf.columns 
    for old_col in cols: 
        new_col = re.sub(r'[^a-zA-Z0-9_]', '', old_col)
        new_col = new_col.upper()
        prepared_sdf = prepared_sdf.rename(F.col(old_col), new_col)

    temp_df = session.table("ML_SNOWPARK_CI_CD_DB.ML_PROCESSING.PROCESSED_INPUT").limit(0)  # Define the table schema without fetching any data.
    final_table = temp_df.natureal_join(prepared_sdf, how='outer').fillna(0)
    final_table = final_table.with_column('TIMESTAMP', F.current_timestamp())

    if final_table.count() > 0:
        count = final_table.count()
        print("\nRows to be written to ML_SNOWPARK_CI_CD_DB.ML_PROCESSING.PROCESSED_INPUT = ", final_table.count())
        final_table.write.mode('append').save_as_table("ML_SNOWPARK_CI_CD.ML_PROCESSING.PROCESSED_INPUT")
    else:
        print("Final DF Empty")
        sys.exit()

    return str(f'{count} rows written to the ML_PROCESSING.PROCESSED_INPUT stream at ' + str(datetime.now()))

sproc_process_input(session)

sproc_de = session.sproc.register(func=sproc_process_input, 
                                  name='sproc_process_input', 
                                  is_permanent=True,
                                  replace=True,
                                  stage_location='@ML_PROCESSING.ML_MODELS',
                                  packages=['snowflake-ml-python', 
                                            'snowflake-snowpark-python']
                                )

