import os
from google.cloud import bigquery
import pandas as pd
import tabulate as tb
import textwrap
import dtale

def display_dataframe(df, tablefmt="grid", max_width=40):
    """
    Displays a pandas DataFrame in a tabular format using the tabulate library.

    Args:
        df (pd.DataFrame): The pandas DataFrame to display.
        tablefmt (str): The table format to use for displaying the DataFrame.
                        Default is "grid".
        max_width (int): The maximum width of each column. Content exceeding this width
                         will be wrapped to the next line.

    Returns:
        None
    """
    def wrap_text(text, max_width):
        wrapper = textwrap.TextWrapper(width=max_width)
        return '\n'.join(wrapper.wrap(text))

    wrapped_df = df.copy()

    for col in wrapped_df.columns:
        wrapped_df[col] = wrapped_df[col].astype(str).apply(lambda x: wrap_text(x, max_width))

    print(tb.tabulate(wrapped_df, headers=wrapped_df.columns, tablefmt=tablefmt, showindex=False))

def set_values_to_binary(value):
    return 1 if value >= 1 else 0


def get_dataframe(csv_file, sql_query):
    # Check if the CSV file exists
    if os.path.exists(csv_file):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Check if the DataFrame is not empty
        if not df.empty:
            return df

    # If the CSV file doesn't exist or is empty, execute the SQL query
    client = bigquery.Client()
    query_job = client.query(sql_query)
    query_result = query_job.result()

    # Convert the query result to a DataFrame
    df = query_result.to_dataframe()

    # Save the DataFrame to a CSV file for future use
    df.to_csv(csv_file, index=False)

    return df

def show_dtale(dataFrame):
    d = dtale.show(dataFrame)
    d.open_browser()
    input("continue:")