import torch
from processes.extract_inferecne_functions import get_labels_tensor_from_data
from utils.general_functions import custom_set_seed, load_dikta_lchaim_data
from processes.train_mlp import train_model, MLPClassifier


def main_training(x_embeddings_train, y_train, x_embeddings_val, y_val):
    custom_set_seed(42)
    classifier_model = MLPClassifier()
    train_model(x_train=x_embeddings_train, y_train=y_train.long(), x_val=x_embeddings_val, y_val=y_val,
                model=classifier_model, learning_rate=1e-4, batch_size=64,
                epochs=10, checkpoints_dir_path='checkpoints_')


if __name__ == '__main__':
    base_emb_folder = 'mlp_inputs'
    dataset_name = 'lchaim'
    model_name = 'longhero'
    embeddings_inputs_folder_path = f'{base_emb_folder}/{dataset_name}/{model_name}'

    train_data, dev_data, test_data = load_dikta_lchaim_data()

    train_embeddings = torch.load(f'{embeddings_inputs_folder_path}/{model_name}_x_train.pt')
    dev_embeddings = torch.load(f'{embeddings_inputs_folder_path}/{model_name}_x_dev.pt')
    test_embeddings = torch.load(f'{embeddings_inputs_folder_path}/{model_name}_x_test.pt')

    y_train = get_labels_tensor_from_data(train_data)
    y_dev = get_labels_tensor_from_data(dev_data)
    y_test = get_labels_tensor_from_data(test_data)

    main_training(x_embeddings_train=train_embeddings, y_train=y_train.long(),
                  x_embeddings_val=dev_embeddings, y_val=y_dev.long())

