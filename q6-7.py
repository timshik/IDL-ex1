import torch

from DatasetLoader import *
from model import SimpleModel
import numpy as np

corona_spike_path = 'corona_spike_sequence.txt'

def get_cov_sequences(path):
    with open(path) as f:
        s = " ".join([l.rstrip() for l in f])
    data = []
    for i in range(len(s)-8):
        data.append(s[i:i+9])
    return data

# get the data
data = get_cov_sequences(corona_spike_path)
data_encoded = np.array(encode_data(data)).astype(np.float64)
tensor_data = torch.Tensor(data_encoded)

# run the data through the model
pre_trained_model = SimpleModel()
pre_trained_model.load('model_weights.ckpt')
outputs = np.ndarray.flatten(pre_trained_model(tensor_data).detach().numpy())

# taking the 5 sequences that scored the best
top5 = np.array(data)[np.argpartition(outputs, -5)[-5:]]

print('notifying the cdc about most detectable peptides')
for peptide in top5:
    print(peptide)
