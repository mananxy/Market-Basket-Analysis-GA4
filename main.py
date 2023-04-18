import os
import pandas as pd
from google.cloud import bigquery
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from helpers import set_values_to_binary,get_dataframe,show_dtale

# Set the path to the JSON credentials file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Auth/retail-ga4-dataset-76ffc0b63205.json'

# Initialize the BigQuery client
client = bigquery.Client()

# Set your project, dataset, and table prefix
project_id = 'bigquery-public-data'
dataset_id = 'ga4_obfuscated_sample_ecommerce'
#table_prefix = 'events_20210131'

# Store results of query in CSV file to avoid re-querying when writing the program.
csv_file="query_result.csv"

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

result = get_dataframe(csv_file,sql_query)

# Split item_categories into separate rows
raw_split_categories = result.assign(item_categories=result["item_categories"].str.split(",")).explode("item_categories")

# raw_split_categories.loc[...]: The .loc[] accessor is used to select rows from the DataFrame based on the boolean conditions within [].
# some of the transactions had items which did not belong to any categories. Removing any such rows where there is no category and where category =New

split_categories = raw_split_categories.loc[(raw_split_categories["item_categories"] != "") & (raw_split_categories["item_categories"] != "New")]



# One-hot encode the item_categories
one_hot_encoded = pd.get_dummies(split_categories, columns=["item_categories"], prefix="", prefix_sep="")

# Group by transaction_id and aggregate the rows
result_encoded_grouped = one_hot_encoded.groupby(["event_date", "transaction_id", "purchased_product_skus"], as_index=False).sum()

result_binary = result_encoded_grouped.copy()
result_binary[result_binary.columns[3:]] = result_encoded_grouped.iloc[:, 3:].applymap(set_values_to_binary)


#Generate frequent item sets that have a support of atleast 10%
# Pass only the one-hot encoded columns to the apriori() function
supported_itemsets = apriori(result_binary.iloc[:, 3:], min_support=0.03, use_colnames=True)
show_dtale(supported_itemsets)
#Generate the rules with their corresponding support, confidence and lift:

rules = association_rules(supported_itemsets, metric="lift", min_threshold=1)
show_dtale(rules)
# Sort the rules by the 'lift' metric in descending order
sorted_rules = rules.sort_values(by='lift', ascending=False)




