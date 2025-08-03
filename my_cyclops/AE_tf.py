import torch
import torch.nn as nn
import math

class SampleTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(SampleTransformerEncoder, self).__init__()
        self.d_model = d_model
        
        # 将每个样本投影到d_model维度
        self.sample_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(d_model, 2)
        
    def forward(self, x):        
        x = self.sample_projection(x)
        x = x.unsqueeze(0)
        encoded = self.transformer(x)
        encoded = encoded.squeeze(0)
        phase_coords = self.output_projection(encoded)
        return phase_coords

class SimpleDecoder(nn.Module):
    def __init__(self, output_dim, dropout=0.1):
        super(SimpleDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, phase_coords):
        return self.decoder(phase_coords)

class PhaseAutoEncoder(nn.Module):
    def __init__(self, input_dim, n_celltypes=0, celltype_embedding_dim=4, dropout=0.1):
        super(PhaseAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.n_celltypes = n_celltypes
        self.use_celltype = n_celltypes > 0
        self.celltype_embedding_dim = celltype_embedding_dim
        
        if self.use_celltype:
            self.celltype_embedding = nn.Embedding(n_celltypes, celltype_embedding_dim)
            self.scale_transform = nn.Linear(celltype_embedding_dim, input_dim)
            self.additive_transform = nn.Linear(celltype_embedding_dim, input_dim)
            self.global_bias = nn.Parameter(torch.zeros(input_dim))
        
        self.encoder = SampleTransformerEncoder(
            input_dim=input_dim,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=dropout
        )
        
        self.decoder = SimpleDecoder(
            output_dim=input_dim,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, celltype_indices=None):
        phase_coords_normalized = self.encode(x, celltype_indices)
        reconstructed = self.decoder(phase_coords_normalized)
        return phase_coords_normalized, reconstructed
    
    def encode(self, x, celltype_indices=None):
        if self.use_celltype and celltype_indices is not None:
            celltype_emb = self.celltype_embedding(celltype_indices)
            scale_factor = self.scale_transform(celltype_emb)
            additive_factor = self.additive_transform(celltype_emb)
            modified_input = x * (1 + scale_factor) + additive_factor + self.global_bias
        else:
            modified_input = x
        
        modified_input = self.dropout(modified_input)
        phase_coords = self.encoder(modified_input)

        norm = torch.norm(phase_coords, dim=1, keepdim=True) + 1e-8
        phase_coords_normalized = phase_coords / norm
        
        return phase_coords_normalized
    
    def decode(self, phase_coords):
        return self.decoder(phase_coords)