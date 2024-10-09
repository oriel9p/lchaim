import json
import random
from typing import Union
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
from transformers import Trainer


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


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data


def evaluate_model(model, test_x, y_test, labels_mapping, batch_size=32, device='cuda'):
    """
    Evaluates the given model on the test data and visualizes the results.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        test_x (torch.Tensor): The input features for testing.
        y_test (torch.Tensor): The ground truth labels for testing.
        labels_mapping (dict): Dictionary mapping labels to indices.
        batch_size (int): The batch size for evaluation. Default is 32.
        device (str): The device to run the model on ('cuda' or 'cpu'). Default is 'cuda'.

    Returns:
        tuple: (predictions, accuracy, report) where:
            - predictions (torch.Tensor): The model's predictions on the test data.
            - accuracy (float): The accuracy of the model on the test data.
            - report (str): The classification report as a string.
    """
    model.eval()  # Set the model to evaluation mode

    # Move model and data to the specified device
    model.to(device)
    test_x = test_x.to(device)
    y_test = y_test.to(device)

    # Create a DataLoader for test data
    test_dataset = TensorDataset(test_x, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to track gradients during evaluation
        for inputs, labels in test_loader:
            outputs = model(inputs)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get predictions
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    all_preds, accuracy, report = evaluate_model_base_on_preds_and_res(y_preds=all_preds, y_test=all_labels,
                                                                       labels_mapping=labels_mapping)

    return all_preds, accuracy, report


def evaluate_model_base_on_preds_and_res(y_preds, y_test, labels_mapping):
    # Calculate accuracy
    correct = (y_preds == y_test).sum().item()
    accuracy = correct / len(y_test)

    # Mapping labels
    labels = [label for label, _ in sorted(labels_mapping.items(), key=lambda item: item[1])]

    # Compute and display confusion matrix
    cm = confusion_matrix(y_test, y_preds, labels=list(labels_mapping.values()))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Compute and display classification report
    report = classification_report(y_test, y_preds, target_names=labels, digits=5)
    print("Classification Report:")
    print(report)

    return y_preds, accuracy, report


def custom_set_seed(seed):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch CPU
    torch.manual_seed(seed)

    # Set seed for PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multiple GPUs

    # Ensure that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dikta_lchaim_data():
    train_data_parquet = pd.read_parquet('../datasets/dikta_heb_datasets/dikta_train_data.parquet')
    control_train_data = train_data_parquet.to_dict(orient='records')
    control_dev_data = read_json('../datasets/dikta_heb_datasets/dikta_dev_data.json')
    control_test_data = read_json('../datasets/dikta_heb_datasets/dikta_test_data.json')
    return control_train_data, control_dev_data, control_test_data


def output_results(trainer: Trainer, dataset: Dataset, test_array, model_name, dataset_name, labels_mapping=None):
    results = trainer.predict(test_dataset=dataset)
    predictions = results.predictions

    if labels_mapping is None:
        labels_mapping = {'c': 0, 'e': 1, 'n': 2}

    print(f'Results for - dataset:{dataset_name}, model: {model_name}')
    res = evaluate_model_base_on_preds_and_res(y_preds=predictions.argmax(axis=1),
                                               y_test=test_array,
                                               labels_mapping=labels_mapping)
    return res


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