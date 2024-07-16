import json
from typing import Union
import pandas as pd
from pandas import DataFrame


def read_jsonl_file(nl_file_path, return_df=True) -> Union[DataFrame, list]:
    """
    :param return_df: return df with 'text', 'key', 'text_translated' columns
    :param nl_file_path:
    :return: list of dicts(keys: 'text', 'key', 'text_translated') per translation
    """
    with open(nl_file_path, 'r') as f:
        jsonl_content = f.read()
        result = [json.loads(jline) for jline in jsonl_content.splitlines()]
    if return_df:
        return pd.DataFrame(result)
    return result


def get_translation_by_text_key(text_key: str, translation_df: dict, text_key_column_name='key'):
    return translation_df[translation_df[text_key_column_name] == text_key][['text_translated']].values.tolist()[0][0]


def read_json_mappings_of_pairs(json_file_path, return_df=True):
    with open(json_file_path, 'r') as f:
        data = json.loads(f.read())
    if return_df:
        return pd.DataFrame(data)
    return data


def extract_translations_from_mappings_df(mappings_df: DataFrame, translation_df) -> DataFrame:
    cloned_mappings = mappings_df.copy()

    cloned_mappings['premise'] = cloned_mappings['premise_id'].apply(
        lambda x: get_translation_by_text_key(x, translation_df=translation_df))
    cloned_mappings['hypothesis'] = cloned_mappings['hypothesis_id'].apply(
        lambda x: get_translation_by_text_key(x, translation_df=translation_df))

    return cloned_mappings


def translate_dataset(translation_jsonl_path, mappings_json_path, return_list_of_dicts=True, specific_columns_list=None):
    translation_df = read_jsonl_file(translation_jsonl_path)
    mappings_df = read_json_mappings_of_pairs(mappings_json_path)
    res = extract_translations_from_mappings_df(mappings_df, translation_df)
    if specific_columns_list:
        res = res[specific_columns_list]
    if return_list_of_dicts:
        return res.to_dict(orient='records')
    return res
