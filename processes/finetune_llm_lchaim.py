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


def preprocess_heblchaim_data_before_ft(lchaim_train_data, lchaim_dev_data, lchaim_test_data,
                                        lchaim_y_train, lchaim_y_dev, lchaim_y_test,
                                        tokenizer, lchaim_context_window):
    lchaim_train_df = pd.DataFrame(lchaim_train_data)
    lchaim_dev_df = pd.DataFrame(lchaim_dev_data)
    lchaim_test_df = pd.DataFrame(lchaim_test_data)

    lchaim_train_df['paires'] = lchaim_train_df['translation1'] + ' [SEP] ' + lchaim_train_df[
        'translation2'] + ' [SEP] '
    lchaim_dev_df['paires'] = lchaim_dev_df['translation1'] + ' [SEP] ' + lchaim_dev_df['translation2'] + ' [SEP] '
    lchaim_test_df['paires'] = lchaim_test_df['translation1'] + ' [SEP] ' + lchaim_test_df['translation2'] + ' [SEP] '

    lchaim_train_pairs = lchaim_train_df['paires'].tolist()
    lchaim_dev_pairs = lchaim_dev_df['paires'].tolist()
    lchaim_test_pairs = lchaim_test_df['paires'].tolist()

    lchaim_tokenized_train = tokenize_one_sample(tokenizer=tokenizer, sample=lchaim_train_pairs,
                                                 max_length=lchaim_context_window)
    lchaim_tokenized_dev = tokenize_one_sample(tokenizer=tokenizer, sample=lchaim_dev_pairs,
                                               max_length=lchaim_context_window)
    lchaim_tokenized_test = tokenize_one_sample(tokenizer=tokenizer, sample=lchaim_test_pairs,
                                                max_length=lchaim_context_window)

    lchaim_train_data = Dataset.from_dict({
        'input_ids': lchaim_tokenized_train['input_ids'],
        'attention_mask': lchaim_tokenized_train['attention_mask'],
        'labels': lchaim_y_train
    })

    lchaim_dev_data = Dataset.from_dict({
        'input_ids': lchaim_tokenized_dev['input_ids'],
        'attention_mask': lchaim_tokenized_dev['attention_mask'],
        'labels': lchaim_y_dev
    })

    lchaim_test_data = Dataset.from_dict({
        'input_ids': lchaim_tokenized_test['input_ids'],
        'attention_mask': lchaim_tokenized_test['attention_mask'],
        'labels': lchaim_y_test
    })

    return {'lchaim_train_data': lchaim_train_data, 'lchaim_dev_data': lchaim_dev_data,
            'lchaim_test_data': lchaim_test_data}


def finetune_llm_on_lchaim(lchaim_train_data, lchaim_dev_data, lchaim_test_data,
                           lchaim_y_train, lchaim_y_dev,lchaim_y_test,
                           model_name, folder_path_to_save_model, hf_model_path_to_load, hf_tokenizer_path_to_load,
                           device, lchaim_context_window, num_labels=3, print_results=True):
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_path_to_load, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path_to_load)
    model.to(device)
    lchaim_processed_data_for_ft = preprocess_heblchaim_data_before_ft(lchaim_train_data=lchaim_train_data,
                                                                       lchaim_dev_data=lchaim_dev_data,
                                                                       lchaim_test_data=lchaim_test_data,
                                                                       lchaim_y_train=lchaim_y_train,
                                                                       lchaim_y_dev=lchaim_y_dev,
                                                                       lchaim_y_test=lchaim_y_test,
                                                                       tokenizer=tokenizer,
                                                                       lchaim_context_window=lchaim_context_window)

    lchaim_train_dataset = lchaim_processed_data_for_ft['lchaim_train_data']
    lchaim_dev_dataset = lchaim_processed_data_for_ft['lchaim_dev_data']
    lchaim_test_dataset = lchaim_processed_data_for_ft['lchaim_test_data']

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-5,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10
    )

    lchaim_trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lchaim_train_dataset,
        eval_dataset=lchaim_dev_dataset,
        compute_metrics=lchaim_test_dataset,
    )

    lchaim_trainer.train()

    model.save_pretrained(folder_path_to_save_model)

    if print_results:
        output_results(trainer=lchaim_trainer, dataset=lchaim_test_data, test_array=lchaim_y_test.numpy(),
                       model_name=model_name, dataset_name='lchaim test data')
