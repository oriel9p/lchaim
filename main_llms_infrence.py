import torch
from transformers import AutoTokenizer, AutoModel
from utils.extract_inferecne_functions import run_and_save_embeddings_per_data_type
from utils.general_functions import load_dikta_lchaim_data


def longhero_hidden_layer_extraction(train_data, dev_data, test_data, model_name,
                                     embeddings_destination_folder_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained('HeNLP/LongHeRo')
    model = AutoModel.from_pretrained('HeNLP/LongHeRo')
    run_and_save_embeddings_per_data_type(train_data=train_data, dev_data=dev_data, test_data=test_data,
                                          model_name=model_name, model=model, tokenizer=tokenizer, device=device,
                                          max_length=4096, inference_batch_size=16,
                                          data_is_df=False, destination_folder_path=embeddings_destination_folder_path)


def alepbert_hidden_layer_extraction(train_data, dev_data, test_data,
                                     model_name, embeddings_destination_folder_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained('onlplab/alephbert-base')
    model = AutoModel.from_pretrained('onlplab/alephbert-base')
    run_and_save_embeddings_per_data_type(train_data=train_data, dev_data=dev_data, test_data=test_data,
                                          model_name=model_name, model=model, tokenizer=tokenizer, device=device,
                                          max_length=512, inference_batch_size=512,
                                          data_is_df=False, destination_folder_path=embeddings_destination_folder_path)


if __name__ == '__main__':
    lchaim_train_data, lchaim_dev_data, lchaim_test_data = load_dikta_lchaim_data()

    hero_model_name = 'longhero'
    longhero_lchim_destination_folder_path = f'mlp_inputs/lchaim/{hero_model_name}'
    longhero_hidden_layer_extraction(train_data=lchaim_train_data, dev_data=lchaim_dev_data, test_data=lchaim_test_data,
                                     embeddings_destination_folder_path=longhero_lchim_destination_folder_path,
                                     model_name=hero_model_name)

    aleph_model_name = 'alephbert'
    alephbert_lchaim_destination_folder_path = f'mlp_inputs/lchaim/{aleph_model_name}'
    longhero_hidden_layer_extraction(train_data=lchaim_train_data, dev_data=lchaim_dev_data, test_data=lchaim_test_data,
                                     embeddings_destination_folder_path=alephbert_lchaim_destination_folder_path,
                                     model_name=aleph_model_name)
