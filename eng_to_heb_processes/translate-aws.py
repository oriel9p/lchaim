# import boto3

# translate = boto3.client('translate', region_name='us-east-1')

# def translate_text(text, source_lang='en', target_lang='he'):
#     result = translate.translate_text(Text=text, 
#                                       SourceLanguageCode=source_lang, 
#                                       TargetLanguageCode=target_lang)
#     return result['TranslatedText']

# # Example of translating a single sentence
# translated_text = translate_text("Hello, world!")
# print(translated_text)


import boto3
import json
import itertools

# Initialize the Translate client
session = boto3.Session(profile_name='lechaim')


# translate = boto3.client('translate', region_name='us-east-1')
translate = session.client('translate', region_name='us-east-1')

def translate_text(text, source_lang='en', target_lang='he'):
    """Translate text from source language to target language."""
    result = translate.translate_text(Text=text, SourceLanguageCode=source_lang, TargetLanguageCode=target_lang)
    return result['TranslatedText']

def translate_jsonl(input_path, output_path):
    """Translate the content of a JSONL file and save the result to a new JSONL file."""
    starting_line = 0
    i = starting_line
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'a', encoding='utf-8') as outfile:
        for line in itertools.islice(infile, starting_line, None):
            stripped_line = line.strip()
            if stripped_line in ['{', '}']: continue
            # Split the text and text_id
            try:
                text, text_id = stripped_line.split('": ')
                text = text.replace('"', '')
                text_id = text_id.replace('"', '')
                # Translate the text
                items = {
                    "text" : text,
                    "key": text_id,
                    "text_translated": translate_text(text)
                }
                outfile.write(json.dumps(items, ensure_ascii=False) + '\n')
                i += 1
                if i % 10  == 0: 
                    print(f'{i} lines translated so far')
            except:
                print(f'Error in line: {stripped_line} - skipping')
            

def load_jsonl(file_path):
    """Load a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Paths for input and output files
input_file_path = './unique_texts.json'
output_file_path = './eng_to_heb_processes/unique_texts_aws_translated.jsonl'

# Translate the JSONL file
translate_jsonl(input_file_path, output_file_path)

# Confirm the translation by loading and displaying the translated content
translated_data = load_jsonl(output_file_path)
print(translated_data[:2])
