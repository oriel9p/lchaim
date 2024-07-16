import json


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


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
    main()
