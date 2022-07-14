import pandas as pd
import numpy as np
import json

def input_to_json(input_):
    new_json = {}

    try:
        if type(input_) == str:
            new_json = json.loads(input_)

        elif type(input_) == list:
            new_json = json.dumps(input_)

    except (TypeError, ValueError):
        new_json = {}

    return new_json


def order_ids_to_json(order_id_col, json_row_org):
    json_row = json_row_org.copy()

    if not json_row:
        return np.nan

    for i in range(len((json_row['purchase']))):
        # Add OrderId to JSON
        json_row['purchase'][i - 1]['OrderId'] = order_id_col
        # Add totalSum to JSON
        json_row['purchase'][i - 1]['totalSumTemp'] = json_row_org['totalSum']

    return json_row


def get_json(value):
    try:
        new_json = json.loads(str(value))
    except:  # JSONDecodeError
        new_json = {}
    return new_json


class JsonFormatter:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def json_formatter(self, json_col_name, multilevel_col_name=None, n_rows=None):
        """
        Observe that if n_rows is selected only a certain number of rows will be selected.
        """
        # Limit N rows to parse
        copy_df = self.df.copy()
        t1 = copy_df[~copy_df[json_col_name].isnull()].reset_index(drop=True)
        n_rows = n_rows if n_rows is not None else t1.shape[0]
        t1 = t1.head(n_rows)
        t1[json_col_name] = t1[json_col_name].apply(input_to_json)

        if multilevel_col_name is not None:
            t1[json_col_name] = t1.apply(lambda x: order_ids_to_json(x['OrderId'], x[json_col_name]),
                                         axis=1)  # this step should be refactored it uses a unique col from
            # transactions
            t2 = pd.json_normalize(t1[json_col_name])

            # Use also   record_path=['purchase'] from pd.json_normalize
            t3 = pd.concat([pd.DataFrame(pd.json_normalize(x)) for x in t2[multilevel_col_name]], ignore_index=True)

            t4 = t3.join(t2)
            t4.drop(multilevel_col_name, axis=1, inplace=True)
            t4['totalSum'] = t4['totalSumTemp']
            t4.drop('totalSumTemp', axis=1, inplace=True)

            return pd.merge(copy_df, t4, how='left', on='OrderId')

        else:
            copy_df[json_col_name] = copy_df[json_col_name].apply(input_to_json)

            return copy_df.join(pd.json_normalize(copy_df[json_col_name]))

    def json_to_df(self, json_col_name):
        copy_df = self.df.copy()
        copy_df[json_col_name + '_json'] = copy_df[json_col_name].apply(get_json)
        json_norm_df = pd.json_normalize(copy_df[json_col_name + '_json'])
        return json_norm_df

    def generate_json_items(self, json_col_name, json_key_level1):
        """

        :param json_col_name: Ex: Purchases
        :param json_key_level1: Ex: purchase
        :return:
        """
        json_pd = self.json_to_df(json_col_name)
        # Convert to pd series
        json_items = json_pd[json_key_level1].apply(pd.Series)

        # For every column I will add item + col as name
        json_items.columns = ['item_' + str(col) for col in json_items.columns]

        return json_items, json_pd

    def extend_json_columns(self, json_col_name, json_key_level1):
        json_col_dfs = []
        nan_value = {'barcode': '', 'epc': '', 'name': '', 'price': np.nan,
                     'currency': '', 'image': '', 'imageUrl': '', 'count': np.nan,
                     'vatValue': np.nan, 'isDisounted': '', 'discountAmount': np.nan,
                     'discountName': '', 'skuId': ''}

        json_items, json_pd = self.generate_json_items(json_col_name, json_key_level1)
        for col in json_items.columns:
            json_items[col] = json_items[col].apply(lambda x: x if x == x else nan_value)
            json_col_df = pd.json_normalize(json_items[col])
            json_col_dfs.append(json_col_df)

        for df in enumerate(json_col_dfs):
            df[1].columns = [col + '_item_' + str(df[0]) for col in df[1].columns]

        transactions_wide = pd.concat([self.df] + [json_pd] + json_col_dfs, axis=1)

        return transactions_wide
