import json
from typing import Union
import pandas as pd
from scipy.stats import truncnorm
import numpy as np


def read_jsonl_file(nl_file_path, return_df=True) -> Union[pd.DataFrame, list]:
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
    try:
        return translation_df[translation_df[text_key_column_name] == text_key][['text_translated']].values.tolist()[0][0]
    except Exception as e:
        print(f'error with key: {text_key}. (Error: {e})')


def read_json_mappings_of_pairs(json_file_path, return_df=True):
    with open(json_file_path, 'r') as f:
        data = json.loads(f.read())
    if return_df:
        return pd.DataFrame(data)
    return data


def extract_translations_from_mappings_df(mappings_df: pd.DataFrame, translation_df) -> pd.DataFrame:
    cloned_mappings = mappings_df.copy()

    cloned_mappings['premise'] = cloned_mappings['premise_id'].apply(
        lambda x: get_translation_by_text_key(x, translation_df=translation_df))
    cloned_mappings['hypothesis'] = cloned_mappings['hypothesis_id'].apply(
        lambda x: get_translation_by_text_key(x, translation_df=translation_df))

    return cloned_mappings


def translate_dataset(translation_jsonl_path, mappings_json_path, return_list_of_dicts=True,
                      specific_columns_list=None):
    translation_df = read_jsonl_file(translation_jsonl_path)
    mappings_df = read_json_mappings_of_pairs(mappings_json_path)
    res = extract_translations_from_mappings_df(mappings_df, translation_df)
    if specific_columns_list:
        res = res[specific_columns_list]
    if return_list_of_dicts:
        return res.to_dict(orient='records')
    return res


# text translations sampling
def load_json_file(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def load_multiple_json_files(file_paths_list, return_df=False):
    data = []
    for p in file_paths_list:
        data += load_json_file(p)
    if return_df:
        return pd.DataFrame(data)
    return data


def extract_premises_text_ids(mappings_files_paths_list):
    data = load_multiple_json_files(mappings_files_paths_list, return_df=False)
    all_premises = [data[i]['premise_id'] for i in range(len(data))]
    premises_text_ids = list(set(all_premises))
    return premises_text_ids


def get_translation_df(translation_jsonl_path):
    translation_df = read_jsonl_file(translation_jsonl_path)
    translation_df['text_len'] = translation_df['text'].apply(len)
    return translation_df


def get_premises_translation_df(translation_df, mappings_files_paths_list):
    premises_text_ids = extract_premises_text_ids(mappings_files_paths_list)
    premises_translation_df = translation_df[translation_df['key'].isin(premises_text_ids)].reset_index(drop=True)
    return premises_translation_df


def get_most_similar_text_lengths_premises_ids_by_distribution(samples_array, premises_translation_df):
    used_ids = set()
    selected_ids = []

    for sample in samples_array:
        # Filter out used IDs from the DataFrame
        available_df = premises_translation_df[~premises_translation_df['key'].isin(used_ids)]

        if available_df.empty:
            break  # Stop if there are no available IDs left

        closest_row = available_df.iloc[(available_df['text_len'] - sample).abs().argsort()[:1]]
        closest_id = closest_row['key'].values[0]

        selected_ids.append(closest_id)
        used_ids.add(closest_id)

    return selected_ids


def sample_premises_text_ids_by_text_length(translation_df, mappings_files_paths_list, num_samples=40,
                                            lower_bound=0, upper_bound=7000, seed=0):
    premises_translation_df = get_premises_translation_df(translation_df=translation_df,
                                                          mappings_files_paths_list=mappings_files_paths_list)
    mean_length = premises_translation_df['text_len'].mean()
    std_dev_length = premises_translation_df['text_len'].std()
    np.random.seed(seed)

    samples = truncnorm.rvs(
        (lower_bound - mean_length) / std_dev_length,
        (upper_bound - mean_length) / std_dev_length,
        loc=mean_length,
        scale=std_dev_length,
        size=num_samples,
    )
    rounded_samples_array = samples.round()
    selected_ids = get_most_similar_text_lengths_premises_ids_by_distribution(samples_array=rounded_samples_array,
                                                                              premises_translation_df=
                                                                              premises_translation_df)
    return selected_ids


def extract_pairs_translations_according_premises_ids(premises_ids_list, pairs_mappings_paths_list, translation_df):
    pairs_data = load_multiple_json_files(pairs_mappings_paths_list, return_df=True)
    chosen_pairs_data = pairs_data[pairs_data.premise_id.isin(premises_ids_list)].reset_index(drop=True)
    premise_df = chosen_pairs_data.merge(
        translation_df,
        left_on='premise_id',
        right_on='key',
        how='left')

    hypothesis_df = chosen_pairs_data.merge(
        translation_df,
        left_on='hypothesis_id',
        right_on='key',
        how='left',
    )

    final_df = pd.DataFrame({
        'premise_text_key': premise_df['key'],
        'uid': premise_df['uid'],
        'english_premise': premise_df['text'],
        'hebrew_premise': premise_df['text_translated'],
        'english_hypothesis': hypothesis_df['text'],
        'hebrew_hypothesis': hypothesis_df['text_translated'],
        'label': chosen_pairs_data['label'],
        'premise_text_len': premise_df['text_len']
    })

    return final_df


def get_sampled_translations_according_to_premises_lengths(translation_jsonl_path,
                                                           mappings_files_paths_list,
                                                           num_samples):
    translation_df = get_translation_df(translation_jsonl_path=translation_jsonl_path)
    samples_premises_ids = sample_premises_text_ids_by_text_length(translation_df=translation_df,
                                                                   mappings_files_paths_list=mappings_files_paths_list,
                                                                   num_samples=num_samples)
    sampled_translated_pairs = extract_pairs_translations_according_premises_ids(premises_ids_list=samples_premises_ids,
                                                                                 pairs_mappings_paths_list=
                                                                                 mappings_files_paths_list,
                                                                                 translation_df=translation_df)

    return sampled_translated_pairs
