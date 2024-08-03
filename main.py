import json
from eng_to_heb_processes import eng_to_heb


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
                       destinations_paths: list[str]):
    for mappings_p, dest_p in zip(jsons_mapping_pairs_paths, destinations_paths):
        print(f'Translating {mappings_p}')
        heb_set = eng_to_heb.translate_dataset(translation_jsonl_path=json_translations_file_path,
                                               mappings_json_path=mappings_p,
                                               return_list_of_dicts=True,
                                               specific_columns_list=['uid', 'premise', 'hypothesis', 'label'])
        # save_json(heb_set, dest_p)
        # print(f'Saved hebrew dataset to {dest_p}. Samples={len(heb_set)}')
        return heb_set


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

    save_json(unique_texts, "unique_texts.json")

    for file in files:
        output_file = f"mapped_{file.replace('.jsonl', '.json')}"
        save_json(mapped_data[file], output_file)


if __name__ == "__main__":
    # main()

    general_jsons_mapping_pairs_paths = ['datasets/mapped_datasets/mapped_train.json',
                                         'datasets/mapped_datasets/mapped_dev.json',
                                         'datasets/mapped_datasets/mapped_test.json'
                                         ]

    dikta_trans_destinations_paths = ['datasets/dikta_heb_datasets/dikta_heb_train.json',
                                      'datasets/dikta_heb_datasets/dikta_heb_dev.json',
                                      'datasets/dikta_heb_datasets/dikta_heb_test.json'
                                      ]
    dikta_json_translations_file_path = 'eng_to_heb_processes/unique_texts.jsonl.dlm2.0translated (1).jsonl'

    aws_trans_destinations_paths = ['datasets/aws_heb_datasets/aws_heb_train.json',
                                    'datasets/aws_heb_datasets/aws_heb_dev.json',
                                    'datasets/aws_heb_datasets/aws_heb_test.json'
                                    ]
    aws_json_translations_file_path = 'eng_to_heb_processes/unique_texts_aws_translated.jsonl'

    translate_datasets(
        jsons_mapping_pairs_paths=general_jsons_mapping_pairs_paths,
        json_translations_file_path=aws_json_translations_file_path,
        destinations_paths=aws_trans_destinations_paths,
    )
