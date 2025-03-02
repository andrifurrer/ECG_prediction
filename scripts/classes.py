import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.transformer import TransformerEncoderLayer


class PreprocessedDataset(Dataset):
    '''
    Defines a Dataset class for individual batch laoding to memory disk
    '''
    def __init__(self, X, y):
        """
        Args:
            X (torch.Tensor): Input features
            y (torch.Tensor): Target labels
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class PositionalEncoding(nn.Module):
    '''
    Defines the positional encoding of the Transformer input to keep track of the position of the sequences, inherits from the nn.Module class
    '''
    def __init__(self, d_model, max_len = 1000):
        super().__init__()  # new version of: super(PositionalEncodingLayer, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        #print("Shape of pe:", pe.size())
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Shape: [max_len, 1], Arange: Returns a 1-D tensor from start to stop, Unsqueeze: Returns a new tensor with a dimension of size one inserted at the specified position
        #print("Shape of position:", position.size())
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # Shape: [d_model // 2]
        #print("Shape of div_term", div_term.size())
        
        # Expand div_term to match the shape of position
        div_term = div_term.unsqueeze(0)  # Shape: [1, d_model // 2]
        div_term = div_term.expand(max_len, -1)  # Shape: [max_len, d_model // 2]

        # Make sure div_term is of shape [max_len, d_model] to broadcast properly
        div_term_full = torch.zeros(max_len, d_model)
        div_term_full[:, 0::2] = div_term  # Fill every other column with div_term
        #print("Corrected shape of div_term", div_term.size())

        pe[:, 0::2] = torch.sin(position * div_term_full[:, 0::2])  # Sine for even indices
        pe[:, 1::2] = torch.cos(position * div_term_full[:, 1::2])  # Cosine for odd indices
        # pe = pe.unsqueeze(0)
        # x = x + pe[:, :x.size(1)]
        
        self.pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        
    def forward(self, x):
        # Add positional encoding to input tensor
        x = x + self.pe[:, :x.size(1), :]
        return x
    

class FullTransformer(nn.Module):
    '''
    Defines the basic structure of a transformer, inherits from the nn.Module class
    '''
    def __init__(self, input_dim, output_dim, num_layers=4, d_model=32, nhead=4, dim_feedforward=128, dropout=0.1):
        super(FullTransformer, self).__init__()
        #self.embedding = nn.Linear(input_dim, d_model) # nn.Linear is a mapping layer to map the input_dim (3) to d_model (32, 64,...)
        self.d_model = d_model
        self.embedding_src = nn.Linear(input_dim, d_model)  # Encoder embedding
        self.embedding_tgt = nn.Linear(1, d_model)         # Decoder embedding
        self.positional_encoding = PositionalEncoding(d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, output_dim)


    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            src: Input sequence (PPG signals) of shape [batch_size, seq_len_src, input_dim]
            tgt: Target sequence (ECG signals) of shape [batch_size, seq_len_tgt, output_dim]
        """
        # Encode the input (PPG signals)
        # Avoids the tracing warning
        if not hasattr(self, 'scale_src'):
            self.scale_src = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=src.device))
        src = self.embedding_src(src) * self.scale_src
        #src = self.embedding_src(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=src.device))
        src = self.positional_encoding(src)
        src = src.transpose(0, 1)  # Transformer expects shape [seq_len, batch_size, d_model]
        
        memory = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Decode the output (ECG signals)
        # Avoids the tracing warning
        if not hasattr(self, 'scale_tgt'):
            self.scale_tgt = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=tgt.device))
        tgt = self.embedding_tgt(tgt) * self.scale_tgt
        #tgt = self.embedding_tgt(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=tgt.device))
        tgt = self.positional_encoding(tgt)
        tgt = tgt.transpose(0, 1)  # Transformer expects shape [seq_len, batch_size, d_model]

        # Ensure tgt_mask and tgt_key_padding_mask do not conflict
        # tgt_mask: [seq_len_tgt, seq_len_tgt]
        # tgt_key_padding_mask: [batch_size, seq_len_tgt]
        output = self.transformer_decoder(
            tgt, memory, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask, 
            memory_key_padding_mask=src_key_padding_mask
        )

        # Map to the output dimension
        output = output.transpose(0, 1)  # Back to shape [batch_size, seq_len_tgt, d_model]
        output = self.fc_out(output)
        return output
    


class Point2PointEncoderTransformer(nn.Module):
    '''
    Defines the basic structure of a transformer, inherits from the nn.Module class
    '''
    def __init__(self, input_dim, output_dim, num_layers=4, d_model=32, nhead=4, dim_feedforward=128, dropout=0.1):
        super(Point2PointEncoderTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model) # nn.Linear is a mapping layer to map the input_dim (3) to d_model (32, 64,...)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.embedding(src) * torch.sqrt(torch.tensor(src.size(-1), dtype=torch.float32))
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = self.fc_out(output)
        return output


class Sequence2PointEncoderTransformer(nn.Module):
    '''
    Defines the basic structure of a transformer, inherits from the nn.Module class
    '''
    def __init__(self, input_dim, output_dim, num_layers=4, d_model=32, nhead=4, dim_feedforward=128, dropout=0.1):
        super(Sequence2PointEncoderTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model) # nn.Linear is a mapping layer to map the input_dim (3) to d_model (32, 64,...)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    
        # Layer Normalization after the transformer (optional)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src):
        src = self.embedding(src)  # Shape: [batch_size, seq_len, d_model]
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)  # Shape: [batch_size, seq_len, d_model]

        # Apply Layer Normalization (optional, depends on whether you want it at this stage)
        output = self.norm(output)

        output = self.fc_out(output)  # Shape: [batch_size, seq_len, output_dim]
        #output = output[:, -1, :].unsqueeze(1)  # Take the last time step and add a dimension
        output = output.mean(dim=1)
        return output  # Shape: [batch_size, 1, output_dim]
    