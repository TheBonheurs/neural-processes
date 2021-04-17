import torch
import torch.nn as nn

class GRUNet(nn.Module):
    """
        Represents a GRU module.

        Parameters
        ----------
        input_dim : int
            Dimension of input layer.

        hidden_dim : int
            Dimension of hidden layer.

        output_dim : int
            Dimension of output layer.

        n_layers : int
            Amount of GRU layers stacked on top of each other.

        drop_prob : int
            Dropout probability p, in the last layer. Default 0.2.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    """
        x : torch.Tensor
            Shape (batch_size, hidden_dim, x_dim)

        h : torch.Tensor
            Shape (x_dim,batch_size, hidden_dim)
    """   
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h
    
    """
        batch_size : int
            Batch size
    """
    def init_hidden(self, batch_size):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden
