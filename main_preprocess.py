import json
from processes import eng_to_heb


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_unique_texts(data):
    unique_texts = {}
    text_id = 0

    for entry in data:
        premise = entry['premise']
        hypothesis = entry['hypothesis']

        if premise not in unique_texts:
            unique_texts[premise] = f"text_{text_id}"
            text_id += 1

        if hypothesis not in unique_texts:
            unique_texts[hypothesis] = f"text_{text_id}"
            text_id += 1

    return unique_texts


def map_translations(data, unique_texts):
    mapped_data = []

    for entry in data:
        mapped_entry = {
            "uid": entry["uid"],
            "premise_id": unique_texts[entry["premise"]],
            "hypothesis_id": unique_texts[entry["hypothesis"]],
            "label": entry["label"]
        }
        mapped_data.append(mapped_entry)

    return mapped_data


def translate_datasets(jsons_mapping_pairs_paths: list[str],
                       json_translations_file_path: str,
                       destinations_paths: list[str],
                       translation_key_name_in_translations_file):

    for mappings_p, dest_p in zip(jsons_mapping_pairs_paths, destinations_paths):
        print(f'Translating {mappings_p}')
        heb_set = eng_to_heb.translate_dataset(translation_jsonl_path=json_translations_file_path,
                                               mappings_json_path=mappings_p,
                                               return_list_of_dicts=True,
                                               specific_columns_list=['uid', 'premise', 'hypothesis', 'label'],
                                               translation_key_column_name=translation_key_name_in_translations_file)
        save_json(heb_set, dest_p)
        print(f'Saved hebrew dataset to {dest_p}. Samples={len(heb_set)}')


def main():
    files = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    all_data = []

    for file in files:
        all_data.extend(load_jsonl(file))

    unique_texts = get_unique_texts(all_data)
    print(f"Number of unique texts: {len(unique_texts)}")
    mapped_data = {}

    for file in files:
        file_data = load_jsonl(file)
        mapped_data[file] = map_translations(file_data, unique_texts)

    save_json(unique_texts, "datasets/mapped_datasets/unique_texts.json")

    for file in files:
        output_file = f"mapped_{file.replace('.jsonl', '.json')}"
        save_json(mapped_data[file], output_file)


if __name__ == "__main__":
    # main()

    general_jsons_mapping_pairs_paths = ['datasets/mapped_datasets/mapped_train.json',
                                         'datasets/mapped_datasets/mapped_dev.json',
                                         'datasets/mapped_datasets/mapped_test.json'
                                         ]

    dikta_trans_destinations_paths = ['datasets/lchaim_dikta_dataset/dikta_heb_train.json',
                                      'datasets/lchaim_dikta_dataset/dikta_heb_dev.json',
                                      'datasets/lchaim_dikta_dataset/dikta_heb_test.json'
                                      ]
    dikta_json_translations_file_path = 'datasets/lchaim_aws_dataset/aws_control_to_to_heb_core_data/unique_texts.jsonl.dlm2.0translated (1).jsonl'

    aws_trans_destinations_paths = ['datasets/lchaim_aws_dataset/aws_heb_train.json',
                                    'datasets/lchaim_aws_dataset/aws_heb_dev.json',
                                    'datasets/lchaim_aws_dataset/aws_heb_test.json'
                                    ]

    eng_lchaim_destinations_paths = [
        'datasets/eng_lchaim/eng_lchaim_train.json',
        'datasets/eng_lchaim/eng_lchaim_dev.json',
        'datasets/eng_lchaim/eng_lchaim_test.json'
    ]
    aws_json_translations_file_path = 'datasets/lchaim_aws_dataset/aws_control_to_to_heb_core_data/unique_texts_aws_translated.jsonl'

    aws_eng_lchaim_file_path = 'datasets/eng_lchaim/reverse_translate_heb_to_eng/heb2eng_unique_texts_aws_translated.jsonl'

    translate_datasets(
        jsons_mapping_pairs_paths=general_jsons_mapping_pairs_paths,
        json_translations_file_path=aws_eng_lchaim_file_path,
        destinations_paths=eng_lchaim_destinations_paths,
        translation_key_name_in_translations_file='text_reverse_translated'
    )
