import torch.nn as nn
import torch.nn.functional as F
import torch

class LinearNet(nn.Module):
    def __init__(self, img_size, hidden_size1, hidden_size2):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(img_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, img_size)

    def forward(self, x):
        x = x.view(-1, 32*32)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def load_model(model, model_dir):
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['net'])

