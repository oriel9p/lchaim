from processes.extract_inferecne_functions import get_labels_tensor_from_data
from utils.general_functions import read_jsonl_file, read_json


def read_heb_snli_data(snli_train_path_file, snli_dev_path_file, snli_test_path_file, snli_labels_mapping=None):
    snli_train_data = read_jsonl_file(snli_train_path_file)
    snli_dev_data = read_jsonl_file(snli_dev_path_file)
    snli_test_data = read_jsonl_file(snli_test_path_file)

    if snli_labels_mapping is None:
        snli_labels_mapping = {'contradiction': 0, 'entailment': 1, 'neutral': 2, '-': 3}

    snli_y_train = get_labels_tensor_from_data(snli_train_data, labels_mapping=snli_labels_mapping,
                                               label_key_name='original_label', data_is_df=True)
    snli_y_dev = get_labels_tensor_from_data(snli_dev_data, labels_mapping=snli_labels_mapping,
                                             label_key_name='original_label', data_is_df=True)
    snli_y_test = get_labels_tensor_from_data(snli_test_data, labels_mapping=snli_labels_mapping,
                                              label_key_name='original_label', data_is_df=True)

    map_cleaned_train_samples = (snli_y_train != 3).numpy()
    map_cleaned_dev_samples = (snli_y_dev != 3).numpy()
    map_cleaned_test_samples = (snli_y_test != 3).numpy()

    snli_train_data = snli_train_data[map_cleaned_train_samples].reset_index(drop=True)
    snli_dev_data = snli_dev_data[map_cleaned_dev_samples].reset_index(drop=True)
    snli_test_data = snli_test_data[map_cleaned_test_samples].reset_index(drop=True)

    snli_y_train = snli_y_train[map_cleaned_train_samples]
    snli_y_dev = snli_y_dev[map_cleaned_dev_samples]
    snli_y_test = snli_y_test[map_cleaned_test_samples]

    hebsnli_data = {'train': (snli_train_data, snli_y_train),
                    'dev': (snli_dev_data, snli_y_dev),
                    'test': (snli_test_data, snli_y_test)
                    }

    return hebsnli_data


def read_lchaim_aws_data(train_path_file, dev_path_file, test_path_file):
    train_data = read_json(train_path_file)
    dev_data = read_json(dev_path_file)
    test_data = read_json(test_path_file)

    uids_to_remove_from_aws_data = ['id_2456', 'id_2457', 'id_2458', 'id_2459']  # text_2851 wasn't translated

    def filter_dicts(list_of_dicts, values_to_exclude, key_name):
        return [d for d in list_of_dicts if d.get(key_name) not in values_to_exclude]

    train_data = filter_dicts(list_of_dicts=train_data, values_to_exclude=uids_to_remove_from_aws_data, key_name='uid')

    y_train = get_labels_tensor_from_data(train_data)
    y_dev = get_labels_tensor_from_data(dev_data)
    y_test = get_labels_tensor_from_data(test_data)

    aws_lchaim_data = {'train': (train_data, y_train),
                    'dev': (dev_data, y_dev),
                    'test': (test_data, y_test)
                    }

    return aws_lchaim_data
