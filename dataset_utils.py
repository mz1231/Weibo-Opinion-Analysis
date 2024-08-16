import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def load_and_upsample_dataset(json_file_path):
    # Load dataset from JSON file
    dataset = load_dataset('json', data_files=json_file_path)
    df = pd.DataFrame(dataset['train'])

    # Splitting the dataframe into separate dataframes based on the labels
    label_1_df = df[df['label'] == 0]
    label_2_df = df[df['label'] == 1]

    len_label_1 = len(label_1_df)  # Length of label 0 dataframe
    len_label_2 = len(label_2_df)  # Length of label 1 dataframe

    # 80% train, 10% val, 10% test
    train_split_ratio = 0.8
    val_split_ratio = 0.1
    test_split_ratio = 0.1

    # For label 0
    label_1_train_size = int(len_label_1 * train_split_ratio)
    label_1_val_size = int(len_label_1 * val_split_ratio)
    label_1_test_size = len_label_1 - label_1_train_size - label_1_val_size

    # For label 1
    label_2_train_size = int(len_label_2 * train_split_ratio)
    label_2_val_size = int(len_label_2 * val_split_ratio)
    label_2_test_size = len_label_2 - label_2_train_size - label_2_val_size

    # Upsample label 2 train data
    label_2_train_upsampled = resample(label_2_df, 
                                      replace=True,     # sample with replacement
                                      n_samples=label_1_train_size,   # to match label 1 train size
                                      random_state=42)   # reproducible results
    print(f"Upsampled Yes: {len(label_2_train_upsampled)}")
    # Concatenate original label 1 train and upsampled label 2 train data
    train_df = pd.concat([label_1_df.iloc[:label_1_train_size], label_2_train_upsampled])

    # Shuffle the train dataframe
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    # Split validation and test datasets
    val_df = pd.concat([label_1_df.iloc[label_1_train_size:label_1_train_size + label_1_val_size],
                        label_2_df.iloc[label_2_train_size:label_2_train_size + label_2_val_size]])
    test_df = pd.concat([label_1_df.iloc[label_1_train_size + label_1_val_size:],
                         label_2_df.iloc[label_2_train_size + label_2_val_size:]])

    # Converting pandas DataFrames into Hugging Face Dataset objects
    dataset_train = Dataset.from_pandas(train_df)
    dataset_val = Dataset.from_pandas(val_df)
    dataset_test = Dataset.from_pandas(test_df)

    # Combine them into a single DatasetDict
    dataset_dict = DatasetDict({
        'train': dataset_train,
        'val': dataset_val,
        'test': dataset_test
    })

    return dataset_dict, test_df, train_df

def load_and_split_dataset(json_file_path):
    # Load dataset from JSON file
    dataset = load_dataset('json', data_files=json_file_path)
    df = pd.DataFrame(dataset['train'])

    # Splitting the dataframe into separate dataframes based on the labels
    label_1_df = df[df['label'] == 0]
    label_2_df = df[df['label'] == 1]

    len_label_1 = len(label_1_df)  # Length of label 0 dataframe
    len_label_2 = len(label_2_df)  # Length of label 1 dataframe

    # Shuffle each label dataframe
    label_1_df = label_1_df.sample(frac=1).reset_index(drop=True)
    label_2_df = label_2_df.sample(frac=1).reset_index(drop=True)

    # 80% train, 10% val, 10% test
    train_split_ratio = 0.8
    val_split_ratio = 0.1
    test_split_ratio = 0.1

    # For label 0
    label_1_train_size = int(len_label_1 * train_split_ratio)
    label_1_val_size = int(len_label_1 * val_split_ratio)
    label_1_test_size = len_label_1 - label_1_train_size - label_1_val_size

    # For label 1
    label_2_train_size = int(len_label_2 * train_split_ratio)
    label_2_val_size = int(len_label_2 * val_split_ratio)
    label_2_test_size = len_label_2 - label_2_train_size - label_2_val_size

    # Split dataset
    label_1_train = label_1_df.iloc[:label_1_train_size]
    label_1_val = label_1_df.iloc[label_1_train_size:label_1_train_size + label_1_val_size]
    label_1_test = label_1_df.iloc[label_1_train_size + label_1_val_size:]

    label_2_train = label_2_df.iloc[:label_2_train_size]
    label_2_val = label_2_df.iloc[label_2_train_size:label_2_train_size + label_2_val_size]
    label_2_test = label_2_df.iloc[label_2_train_size + label_2_val_size:]

    print(f"Yes train: {len(label_2_train)}")
    print(f"Yes val: {len(label_2_val)}")
    print(f"Yes test: {len(label_2_test)}")

    # Concatenating the splits back together
    train_df = pd.concat([label_1_train, label_2_train]).sample(frac=1).reset_index(drop=True)
    val_df = pd.concat([label_1_val, label_2_val]).sample(frac=1).reset_index(drop=True)
    test_df = pd.concat([label_1_test, label_2_test]).sample(frac=1).reset_index(drop=True)

    # Converting pandas DataFrames into Hugging Face Dataset objects
    dataset_train = Dataset.from_pandas(train_df)
    dataset_val = Dataset.from_pandas(val_df)
    dataset_test = Dataset.from_pandas(test_df)

    print(f"Train dataset length: {len(dataset_train)}")
    print(f"Val dataset length: {len(dataset_val)}")
    print(f"Test dataset length: {len(dataset_test)}")

    # Combine them into a single DatasetDict
    dataset = DatasetDict({
        'train': dataset_train,
        'val': dataset_val,
        'test': dataset_test
    })

    return dataset, test_df, train_df

# def upsample_dataset_no_val(json_file_path):
#     # Load dataset from JSON file
#     dataset = load_dataset('json', data_files=json_file_path)
#     df = pd.DataFrame(dataset['train'])

#     # Splitting the dataframe into separate dataframes based on the labels
#     label_1_df = df[df['label'] == 0]
#     label_2_df = df[df['label'] == 1]

#     len_label_1 = len(label_1_df)  # Length of label 0 dataframe
#     len_label_2 = len(label_2_df)  # Length of label 1 dataframe

#     # 80% train, 20% test
#     train_split_ratio = 0.8
#     test_split_ratio = 0.2

#     # For label 0
#     label_1_train_size = int(len_label_1 * train_split_ratio)
#     label_1_test_size = len_label_1 - label_1_train_size

#     # For label 1
#     label_2_train_size = int(len_label_2 * train_split_ratio)
#     label_2_test_size = len_label_2 - label_2_train_size

#     # Upsample label 2 train data
#     label_2_train_upsampled = resample(label_2_df, 
#                                       replace=True,     # sample with replacement
#                                       n_samples=label_1_train_size,   # to match label 1 train size
#                                       random_state=42)   # reproducible results
#     print(f"Upsampled Yes: {len(label_2_train_upsampled)}")

#     # Concatenate original label 1 train and upsampled label 2 train data
#     train_df = pd.concat([label_1_df.iloc[:label_1_train_size], label_2_train_upsampled])

#     # Shuffle the train dataframe
#     train_df = train_df.sample(frac=1).reset_index(drop=True)

#     # Split test dataset
#     test_df = pd.concat([label_1_df.iloc[label_1_train_size:], label_2_df.iloc[label_2_train_size:]])

#     # Converting pandas DataFrames into Hugging Face Dataset objects
#     dataset_train = Dataset.from_pandas(train_df)
#     dataset_test = Dataset.from_pandas(test_df)

#     # Combine them into a single DatasetDict
#     dataset_dict = DatasetDict({
#         'train': dataset_train,
#         'test': dataset_test
#     })

#     print(f"Train Samples: {len(train_df)}")
#     print(f"Test Samples: {len(test_df)}")
    
#     return dataset_dict, test_df, train_df


# def upsample_dataset_no_val(json_file_path):
#     # Load dataset from JSON file
#     dataset = load_dataset('json', data_files=json_file_path)
#     df = pd.DataFrame(dataset['train'])

#     # Splitting the dataframe into separate dataframes based on the labels
#     label_0_df = df[df['label'] == 0]
#     label_1_df = df[df['label'] == 1]

#     # 80% train, 20% test
#     label_0_train, label_0_test = train_test_split(label_0_df, test_size=0.2, random_state=42)
#     label_1_train, label_1_test = train_test_split(label_1_df, test_size=0.2, random_state=42)

#     # Upsample label 1 train data to match label 0 train size
#     label_1_train_upsampled = resample(label_1_train, 
#                                       replace=True,     # sample with replacement
#                                       n_samples=len(label_0_train),   # to match label 0 train size
#                                       random_state=42)   # reproducible results
#     print(f"Upsampled Yes: {len(label_1_train_upsampled)}")

#     # Concatenate original label 0 train and upsampled label 1 train data
#     train_df = pd.concat([label_0_train, label_1_train_upsampled])

#     # Shuffle the train dataframe
#     train_df = train_df.sample(frac=1).reset_index(drop=True)

#     # Concatenate original label 0 and label 1 test data
#     test_df = pd.concat([label_0_test, label_1_test])

#     # Converting pandas DataFrames into Hugging Face Dataset objects
#     dataset_train = Dataset.from_pandas(train_df)
#     dataset_test = Dataset.from_pandas(test_df)

#     # Combine them into a single DatasetDict
#     dataset_dict = DatasetDict({
#         'train': dataset_train,
#         'test': dataset_test
#     })

#     print(f"Train Samples: {len(train_df)}")
#     print(f"Test Samples: {len(test_df)}")
    
#     return dataset_dict, test_df, train_df

def upsample_dataset_no_val(json_file_path):
    # Load dataset from JSON file
    dataset = load_dataset('json', data_files=json_file_path)
    df = pd.DataFrame(dataset['train'])

    # Shuffle the dataframe to ensure randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Splitting the dataframe into separate dataframes based on the labels
    label_1_df = df[df['label'] == 0]
    label_2_df = df[df['label'] == 1]

    len_label_1 = len(label_1_df)  # Length of label 0 dataframe
    len_label_2 = len(label_2_df)  # Length of label 1 dataframe

    # 80% train, 20% test
    train_split_ratio = 0.8
    test_split_ratio = 0.2

    # For label 0
    label_1_train_size = int(len_label_1 * train_split_ratio)
    label_1_test_size = len_label_1 - label_1_train_size

    # For label 1
    label_2_train_size = int(len_label_2 * train_split_ratio)
    label_2_test_size = len_label_2 - label_2_train_size

    # Upsample label 2 train data
    label_2_train_upsampled = resample(label_2_df.iloc[:label_2_train_size], 
                                      replace=True,     # sample with replacement
                                      n_samples=label_1_train_size*6,   # to match label 1 train size
                                      random_state=42)   # reproducible results
    print(f"Upsampled Yes: {len(label_2_train_upsampled)}")

    # Concatenate original label 1 train and upsampled label 2 train data
    train_df = pd.concat([label_1_df.iloc[:label_1_train_size], label_2_train_upsampled])

    # Shuffle the train dataframe
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split test dataset
    test_df = pd.concat([label_1_df.iloc[label_1_train_size:], label_2_df.iloc[label_2_train_size:]])

    # Converting pandas DataFrames into Hugging Face Dataset objects
    dataset_train = Dataset.from_pandas(train_df)
    dataset_test = Dataset.from_pandas(test_df)

    # Combine them into a single DatasetDict
    dataset_dict = DatasetDict({
        'train': dataset_train,
        'test': dataset_test
    })

    print(f"Train Samples: {len(train_df)}")
    print(f"Test Samples: {len(test_df)}")
    
    return dataset_dict, test_df, train_df