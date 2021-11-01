# model
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    """
    very simple model, to be trained on cpu, for code testing.
    """
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.fc1 = nn.Linear(180, 50)
        self.fc2 = nn.Linear(50, 1)
        #self.fc3 = nn.Linear(100, 10)
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout()



    def forward(self, x):

        x = self.dropout1((F.relu(self.fc1(x))))
        #x = self.dropout1(F.relu(self.fc2(x)))

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc2(x))

        return x

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])

