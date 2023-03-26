import os
import pandas as pd
from google.cloud import bigquery
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# Set the path to the JSON credentials file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Auth/retail-ga4-dataset-76ffc0b63205.json'

# Initialize the BigQuery client
client = bigquery.Client()

# Set your project, dataset, and table prefix
project_id = 'bigquery-public-data'
dataset_id = 'ga4_obfuscated_sample_ecommerce'
table_prefix = 'events_20210131'

# Write  SQL query using the wildcard table feature
# Write your SQL query using the wildcard table feature and date range filter
sql_query = f"""
WITH transactions AS (
  SELECT
    event_date,
    event_bundle_sequence_id,
    event_parameters.key AS event_key,
    event_parameters.value.string_value AS event_value
  FROM
    `{project_id}.{dataset_id}.events_*`,
    UNNEST(event_params) AS event_parameters
  WHERE
    event_name = 'purchase'
    AND event_parameters.key = 'transaction_id'
    AND _TABLE_SUFFIX BETWEEN '20210101' AND '20210630'
)

,items AS (
  SELECT
    event_date,
    event_bundle_sequence_id,
    items.item_id AS purchased_product_skus,
    items.item_category AS item_categories
  FROM
    `{project_id}.{dataset_id}.events_*`,
    UNNEST(items) AS items
  WHERE
    _TABLE_SUFFIX BETWEEN '20210101' AND '20210630'
)

,joined_data AS (
  SELECT
    t.event_date,
    t.event_value AS transaction_id,
    i.purchased_product_skus,
    i.item_categories
  FROM
    transactions t
  JOIN
    items i
  ON
    t.event_date = i.event_date
    AND t.event_bundle_sequence_id = i.event_bundle_sequence_id
)

SELECT
  event_date,
  transaction_id,
  STRING_AGG(purchased_product_skus, ',') AS purchased_product_skus,
  STRING_AGG(item_categories, ',') AS item_categories
FROM
  joined_data
GROUP BY
  event_date,
  transaction_id
ORDER BY
  event_date
"""
new_query = True
if new_query:
    query_job = client.query(sql_query)
    result = query_job.to_dataframe()

    # Save the results to a CSV file
    result.to_csv('query_results.csv', index=False)
else:
    # Read the data from the saved CSV file
    result = pd.read_csv('query_results.csv')


# Prepare data for market basket analysis
transactions = result['item_categories'].apply(lambda x: x.split(','))

# Encode transactions for Apriori algorithm
encoder = TransactionEncoder()
encoded_transactions = encoder.fit_transform(transactions)
transaction_df = pd.DataFrame(encoded_transactions, columns=encoder.columns_)

# Perform market basket analysis using the Apriori algorithm
min_support = 0.01  # Adjust this value based on your dataset and requirements
frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)

# Print the results
print(frequent_itemsets)
