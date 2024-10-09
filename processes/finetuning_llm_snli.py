import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from processes.extract_inferecne_functions import tokenize_one_sample
from utils.general_functions import output_results


def compute_accuracy(labels, preds):
    # Convert logits to predicted class (0 or 1)
    pred_labels = np.argmax(preds, axis=1)
    # Calculate accuracy
    accuracy = (labels == pred_labels).mean()
    return {"accuracy": accuracy}


# Function to compute metrics for Hugging Face Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return compute_accuracy(labels, logits)


def preprocess_hebsnli_data_before_ft(snli_train_data, snli_dev_data, snli_test_data,
                                      snli_y_train, snli_y_dev, snli_y_test,
                                      tokenizer, snli_context_window):
    snli_train_df = pd.DataFrame(snli_train_data)
    snli_dev_df = pd.DataFrame(snli_dev_data)
    snli_test_df = pd.DataFrame(snli_test_data)

    snli_train_df['paires'] = snli_train_df['translation1'] + ' [SEP] ' + snli_train_df['translation2'] + ' [SEP] '
    snli_dev_df['paires'] = snli_dev_df['translation1'] + ' [SEP] ' + snli_dev_df['translation2'] + ' [SEP] '
    snli_test_df['paires'] = snli_test_df['translation1'] + ' [SEP] ' + snli_test_df['translation2'] + ' [SEP] '

    snli_train_pairs = snli_train_df['paires'].tolist()
    snli_dev_pairs = snli_dev_df['paires'].tolist()
    snli_test_pairs = snli_test_df['paires'].tolist()

    snli_tokenized_train = tokenize_one_sample(tokenizer=tokenizer, sample=snli_train_pairs,
                                               max_length=snli_context_window)
    snli_tokenized_dev = tokenize_one_sample(tokenizer=tokenizer, sample=snli_dev_pairs,
                                             max_length=snli_context_window)
    snli_tokenized_test = tokenize_one_sample(tokenizer=tokenizer, sample=snli_test_pairs,
                                              max_length=snli_context_window)

    snli_train_data = Dataset.from_dict({
        'input_ids': snli_tokenized_train['input_ids'],
        'attention_mask': snli_tokenized_train['attention_mask'],
        'labels': snli_y_train
    })

    snli_dev_data = Dataset.from_dict({
        'input_ids': snli_tokenized_dev['input_ids'],
        'attention_mask': snli_tokenized_dev['attention_mask'],
        'labels': snli_y_dev
    })

    snli_test_data = Dataset.from_dict({
        'input_ids': snli_tokenized_test['input_ids'],
        'attention_mask': snli_tokenized_test['attention_mask'],
        'labels': snli_y_test
    })

    return {'snli_train_data': snli_train_data, 'snli_dev_data': snli_dev_data, 'snli_test_data': snli_test_data}


def finetune_llm_on_snli(snli_train_data, snli_dev_data, snli_test_data, snli_y_train, snli_y_dev, snli_y_test,
                         model_name, folder_path_to_save_model, hf_model_path_to_load, hf_tokenizer_path_to_load,
                         device, snli_context_window, num_labels=3, print_results=True):
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_path_to_load, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path_to_load)
    model.to(device)
    snli_processed_data_for_ft = preprocess_hebsnli_data_before_ft(snli_train_data=snli_train_data,
                                                                   snli_dev_data=snli_dev_data,
                                                                   snli_test_data=snli_test_data,
                                                                   snli_y_train=snli_y_train,
                                                                   snli_y_dev=snli_y_dev,
                                                                   snli_y_test=snli_y_test,
                                                                   tokenizer=tokenizer,
                                                                   snli_context_window=snli_context_window)

    snli_train_dataset = snli_processed_data_for_ft['snli_train_data']
    snli_dev_dataset = snli_processed_data_for_ft['snli_dev_data']
    snli_test_dataset = snli_processed_data_for_ft['snli_test_data']

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=1e-5,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10
    )

    snli_trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=snli_train_dataset,
        eval_dataset=snli_dev_dataset,
        compute_metrics=snli_test_dataset,
    )

    snli_trainer.train()

    model.save_pretrained(folder_path_to_save_model)

    if print_results:
        output_results(trainer=snli_trainer, dataset=snli_test_data, test_array=snli_y_test.numpy(),
                       model_name=model_name, dataset_name='snli test data')

