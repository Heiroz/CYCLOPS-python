import torch
import torch.nn as nn

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
            encoder_input_dim = input_dim
        else:
            encoder_input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
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
        # add noise when training
        noise = torch.randn_like(phase_coords_normalized) * 0.01
        phase_coords_normalized += noise
        return phase_coords_normalized
    
    def decode(self, phase_coords):
        return self.decoder(phase_coords)