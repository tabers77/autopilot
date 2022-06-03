import os
import pandas as pd

# DATABASES
#from google.cloud import bigquery
import snowflake.connector
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from tuiconns import DbManager
from tuiautopilotml.decorators import time_performance_decor, gc_collect_decor


@time_performance_decor
@gc_collect_decor
def extract_data_to_df(mode, sql_file_location=None, new_dataframe_title='current_dataframe',
                       type_of_connection=None, save_to_csv=False, use_tap=True, data_source_name=None,  *args, **kwargs):
    """
    Example:
        mode: choose from list ['from_database', 'from_csv']
        sql_file_location: path where your sql file is located
        new_dataframe_title:
        data_source_name:
        type_of_connection:
        save_to_csv:
        use_tap:
        *args:
        **kwargs:

    Returns:

    """
    dataframe = pd.DataFrame()

    if mode == 'from_database':
        # Get query
        with open(sql_file_location) as sql_file:
            query = sql_file.read()

        if type_of_connection == 'snowflake':
            print('Obtaining data from snowflake...')

            # Define connector
            if use_tap:
                database = DbManager().get_database(data_source_name)
                engine = database.create_engine()
                connector = engine.connect()
            else:
                connector = snowflake.connector.connect(*args, **kwargs)

            dataframe = pd.read_sql(sql=query, con=connector)
        if save_to_csv:
            print('Saving to csv...')
            dataframe.to_csv(f'{new_dataframe_title}.csv')
            print('Ready')

        return dataframe

    elif mode == 'from_csv':
        print('Loading from csv...')
        dataframe = pd.read_csv(f'{new_dataframe_title}.csv', index_col=0)
        print('Ready')
        return dataframe


@time_performance_decor
@gc_collect_decor
def from_pandas_to_snowflake(df: pd.DataFrame, engine_config=None, use_tap=True,
                             data_source_name=None, table_name='table_name'):
    """
    Saves a pandas df to snowflake

    Examples:
        account = 'vu66182.eu-central-1',
        user = 'your_email_adress' ,
        database = 'SDX04_DB_CDS_DATASCIENCE',
        schema = 'your_dev_env',
        warehouse= 'SDX04_WH_CDS_DATASCIENCE',
        role = SDX04_ROLE_CDS_DATASCIENCE_ANALYST ,
        authenticator='externalbrowser'
    """

    # Specify qmark or numeric to change bind variable formats for server side binding.

    snowflake.connector.paramstyle = 'qmark'  # This let's you append more that 16000 rows
    if use_tap:
        database = DbManager().get_database(data_source_name)
        engine = database.create_engine()

    else:
        url = URL(
            account=engine_config['account'],
            user=engine_config['user'],
            database=engine_config['database'],
            schema=engine_config['schema'],
            warehouse=engine_config['warehouse'],
            role=engine_config['role'],
            authenticator=engine_config['authenticator'])

        df = df.copy()
        engine = create_engine(url)

    connection = engine.connect()

    df.to_sql(table_name, engine, if_exists='replace', index=False, index_label=None)

    connection.close()
    engine.dispose()
