from torch import nn
from torch.nn import functional as F
import torch


class AnnForBinaryClassification(nn.Module):

      def __init__(self, input_features):
          super(AnnForBinaryClassification, self).__init__()
          self.fc1 = nn.Linear(input_features, 64)
          self.fc2 = nn.Linear(64, 32)
          self.fc3 = nn.Linear(32, 1) 

      def forward(self, x):
          x = torch.relu(self.fc1(x))
          x = torch.relu(self.fc2(x))
          x = torch.sigmoid(self.fc3(x))  # Apply sigmoid activation for binary classification
          return x

  