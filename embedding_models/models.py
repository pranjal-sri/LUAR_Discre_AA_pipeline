import torch
import torch.nn as nn

class DeepEmbeddingLUAR(nn.Module):
  def __init__(self, luar_input_dim = 512, discre_input_dim = 845, output_dim = 512, attention_output_dim = 768, positional_info = 'learned'):
    super(DeepEmbeddingLUAR, self).__init__()
    self.layernorm = nn.LayerNorm(luar_input_dim)
    self.fc1 = nn.Linear(luar_input_dim, output_dim)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(output_dim, output_dim)
    self.tanh = nn.Tanh()

  def forward(self, luar_embedding, discre_embeddings):
    luar_embedding = self.layernorm(luar_embedding)
    x = self.relu(self.fc1(luar_embedding))
    x = self.fc2(x)
    x = self.tanh(x)
    return x
  
class DeepEmbeddingModelAvg(nn.Module):
  def __init__(self, luar_input_dim = 512, discre_input_dim = 845, output_dim = 512):
    super(DeepEmbeddingModelAvg, self).__init__()
    self.luar_embeddings_dim = luar_input_dim
    self.discre_embeddings_dim = discre_input_dim
    
    self.layernorm1 = nn.LayerNorm(luar_input_dim)
    self.layernorm2 = nn.LayerNorm(discre_input_dim)
    self.fc1 = nn.Linear(luar_input_dim + discre_input_dim, 768)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(768, output_dim)
    self.tanh = nn.Tanh()



  def forward(self, luar_embedding, discre_embeddings):
    luar_embedding = self.layernorm1(luar_embedding)
    discre_embeddings = self.layernorm2(discre_embeddings)
    x = torch.cat((luar_embedding, discre_embeddings), dim = -1)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    x = self.tanh(x)
    return x