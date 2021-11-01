# create dataset
from torch.utils.data import Dataset
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


num_of_aminos = 20
peptide_length = 9
amino_vocab = {
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'C': 4,
    'Q': 5,
    'E': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19
}

def peptide_to_one_hot(peptide):
    peptide_one_hot = []
    for amino in peptide:
        peptide_one_hot += (amino_to_one_hot(amino))
    return peptide_one_hot


def amino_to_one_hot(amino):
    amino_one_hot = [0]*num_of_aminos
    amino_one_hot[amino_vocab[amino]] = 1
    return amino_one_hot


def encode_data(data):
    encoded_data = []
    for peptide in data:
        encoded_data.append(peptide_to_one_hot(peptide))
    return encoded_data


def read_file(file_name):
    with open(file_name) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


class ProcessData:
    """ex1 dataset."""

    def __init__(self, path_pos, path_neg):
        self.num_of_positive = 0
        self.num_of_negative = 0
        self.data = []
        self.labels = []
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.get_dataset_as_array(path_pos, path_neg)

    def __str__(self):
        return f" number of samples in the data: {len(self.data)} \
            number of positive samples in the data: {self.num_of_positive} ({self.num_of_positive/len(self.data)})%\
                number of negative samples in the data: {self.num_of_negative} ({self.num_of_negative/len(self.data)})%"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def split_train_test(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(self.data, self.labels, test_size=0.1, random_state=1)

    def get_train_data(self):
      return np.array(list(zip(self.X_train, self.Y_train)), dtype=object)

    def get_test_data(self):
        return np.array(list(zip(self.X_test, self.Y_test)), dtype=object)

    def shuffle(self):
        p = np.random.permutation(len(self.data))
        self.labels = self.labels[p]
        self.data = self.data[p]


    def get_dataset_as_array(self, path_pos, path_neg):
        pos = read_file(path_pos)
        neg = read_file(path_neg)
        self.num_of_positive = len(pos)
        self.num_of_negative = len(neg)
        data_encoded = np.array(encode_data(pos+neg)).astype(np.float64)
        labels = np.array([1]*(len(pos)) + [0]*(len(neg)))
        #for 2 outputs
        #labels = np.array([[1,0]]*(len(pos)) + [[0,1]]*(len(neg)))
        self.data = data_encoded
        self.labels = labels


def label_names():
    return {0: 'negative', 1: 'positive'}


class MyDataset(Dataset):
    def __init__(self, the_list):
        self.the_list = the_list

    def __len__(self):
        return len(self.the_list)

    def __getitem__(self, idx):
        item = self.the_list[idx]
        data, label = item
        item = (data, label)
        return item


