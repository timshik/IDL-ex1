import torch
from torch.autograd import Variable
from model import SimpleModel
import numpy as np
import torch.nn as nn

# load the model
pre_trained_model = SimpleModel()
pre_trained_model.load('model_weights.ckpt')

reversed_amino_vocab = amino_vocab = {
    0: 'A',
    1: 'R',
    2: 'N',
    3: 'D',
    4: 'C',
    5: 'Q',
    6: 'E',
    7: 'G',
    8: 'H',
    9: 'I',
    10: 'L',
    11: 'K',
    12: 'M',
    13: 'F',
    14: 'P',
    15: 'S',
    16: 'T',
    17: 'W',
    18: 'Y',
    19: 'V'
}


# find the best input possible for the net
def find_perfect_input(model, input, learning_rate, num_steps=5):
    loss = nn.BCELoss()
    target = Variable(torch.tensor([1.]), requires_grad=False)
    optimizer = torch.optim.Adam([input], lr=learning_rate)
    for i in range(num_steps):
        softmax_input = torch.cat([torch.softmax(input[i:i + 20], dim=0) for i in range(0, 180, 20)])
        output = model.forward(softmax_input)
        # print(output.item())
        optimizer.zero_grad()
        loss_cal = loss(output, target)
        loss_cal.backward()
        optimizer.step()
    return softmax_input.detach().numpy()


# map the input to the peptides space (by creating one hot for every subset of length 20 of the input, where the '1' corresponds to the max value of this subset)
def convert_input_to_peptide(input):
    print(input[0:20])
    peptide = ''
    for i in range(9):
        temp = input[i * 20:(i + 1) * 20]
        hot = np.argmax(temp)
        amino = np.zeros(20)
        amino[hot] = 1
        peptide += reversed_amino_vocab[hot]
        input[i * 20:(i + 1) * 20] = amino
    return input, peptide


def optimize_sequence(input):
    input, peptide = convert_input_to_peptide(find_perfect_input(pre_trained_model, input, 0.1, 5000))
    output = pre_trained_model.forward(torch.tensor(input, requires_grad=False))
    print(f' the peptide is {peptide} and its score in the model is: {output.item()}')


input = torch.randn(180, requires_grad=True)
optimize_sequence(input)
