import pandas as pd
import torch
from tqdm import tqdm


def extract_embeddings_from_sample(model, tokenizer, sample, device, max_length=4096):
    inputs = tokenizer(sample, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).to(
        device)

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu()
    return cls_embedding


def extract_embeddings_per_premise_hypothesis(model, tokenizer, device, data,
                                              data_is_df=False,
                                              premise_col_name='premise',
                                              hypothesis_col_name='hypothesis',
                                              inference_batch_size=10, max_lenght=4096,
                                              ):
    model.to(device)

    embeddings_ = []

    for i in tqdm(range(0, len(data), inference_batch_size)):
        if not data_is_df:
            batched_data = data[i:i + inference_batch_size]
            batch_df = pd.DataFrame(batched_data)
        else:
            batch_df = data.iloc[i:i + inference_batch_size]

        batch_cat_x = batch_df[premise_col_name] + ' [SEP] ' + batch_df[hypothesis_col_name] + ' [SEP] '
        cls_emb = extract_embeddings_from_sample(model=model, tokenizer=tokenizer,
                                                 device=device, sample=batch_cat_x.tolist(),
                                                 max_length=max_lenght)

        embeddings_.append(cls_emb)

        # clean cuda mem
        del batched_data, batch_df, batch_cat_x, cls_emb
        torch.cuda.empty_cache()

    return torch.cat(embeddings_, dim=0)


def run_and_save_embeddings_per_data_type(train_data, dev_data, test_data,
                                          model_name, model, tokenizer, device, destination_folder_path,
                                          max_length=4096, inference_batch_size=10, data_is_df=False,
                                          premise_col_name='premise', hypothesis_col_name='hypothesis',
                                          ):
    print(f'Extracting dataset embeddings by {model_name} model')

    print(f'Extracting train...')
    train_embeddings = extract_embeddings_per_premise_hypothesis(model=model, tokenizer=tokenizer, device=device,
                                                                 data=train_data, max_lenght=max_length,
                                                                 inference_batch_size=inference_batch_size,
                                                                 data_is_df=data_is_df,
                                                                 premise_col_name=premise_col_name,
                                                                 hypothesis_col_name=hypothesis_col_name)
    path_train = f'{destination_folder_path}/{model_name}_x_train.pt'
    torch.save(train_embeddings, path_train)
    print(f'Saved train x ({path_train})')

    print(f'Extracting dev...')
    dev_embeddings = extract_embeddings_per_premise_hypothesis(model=model, tokenizer=tokenizer, device=device,
                                                               data=dev_data, max_lenght=max_length,
                                                               inference_batch_size=inference_batch_size,
                                                               data_is_df=data_is_df,
                                                               premise_col_name=premise_col_name,
                                                               hypothesis_col_name=hypothesis_col_name)
    path_dev = f'{destination_folder_path}/{model_name}_x_dev.pt'
    torch.save(dev_embeddings, path_dev)
    print(f'Saved dev x ({path_dev})')

    print(f'Extracting test...')
    test_embeddings = extract_embeddings_per_premise_hypothesis(model=model, tokenizer=tokenizer, device=device,
                                                                data=test_data, max_lenght=max_length,
                                                                inference_batch_size=inference_batch_size,
                                                                data_is_df=data_is_df,
                                                                premise_col_name=premise_col_name,
                                                                hypothesis_col_name=hypothesis_col_name)
    path_test = f'{destination_folder_path}/{model_name}_x_test.pt'
    torch.save(test_embeddings, path_test)
    print(f'Saved test x ({path_test})')


def get_labels_tensor_from_data(data, label_key_name='label', labels_mapping=None, data_is_df=False):
    if labels_mapping is None:
        labels_mapping = {'c': 0, 'e': 1, 'n': 2}
    if data_is_df:
        str_labels = data[label_key_name].tolist()
    else:
        str_labels = [sample_[label_key_name] for sample_ in data]
    y = torch.tensor([labels_mapping[str_label.lower()] for str_label in str_labels], dtype=torch.int64)
    return y
