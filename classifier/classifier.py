import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Embedding
from classifier.binary_classifier import BinaryClassification

class Classifier(nn.Module):
  def __init__(self, d_model, seq_len, nhead, dim_feedforward, nlayers, device, dropout = 0.5):
    super(Classifier, self).__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.nhead = nhead
    self.dim_feedforward = dim_feedforward
    self.nlayers = nlayers
    self.device = device
    #self.pos_encoder = PositionalEncoding(d_model, dropout)
    self.position_embedding = Embedding(seq_len, d_model)
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
    self.encoder = TransformerEncoder(encoder_layer, nlayers)
    self.binary_classifier = BinaryClassification(seq_len * d_model, device)


  def forward(self, src: Tensor) -> Tensor:
    """
    Args:
        src: Tensor, shape [seq_len, batch_size]
        src_mask: Tensor, shape [seq_len, seq_len]
    Returns:
        output Tensor of shape [seq_len, batch_size, ntoken]
    """
    N, seq_length, embed_size = src.shape
    positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

    src_ = src + self.position_embedding(positions)
    output1 = self.encoder(src_)
    # print(output1.shape)
    # print(output1)
    output = self.binary_classifier(torch.reshape(output1, (N, seq_length * embed_size)))

    return output, output1