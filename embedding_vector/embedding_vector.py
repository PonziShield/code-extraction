import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmbeddingVector(nn.Module):
    def __init__(self):
        super(EmbeddingVector, self).__init__()

        self.embedding_layer = nn.Linear(1, 8)

    def forward(self, x):
        embedded_vector   = F.relu(self.embedding_layer(x))
        final_output  = embedded_vector
        return final_output

