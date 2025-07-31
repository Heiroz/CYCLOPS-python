import torch
import torch.nn as nn
import numpy as np

class PhaseAutoEncoder(nn.Module):
    def __init__(self, input_dim, n_celltypes=0, celltype_embedding_dim=4, 
                 dropout=0.2, phase_expansion_factor=1.0, adaptive_expansion=True):
        super(PhaseAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.n_celltypes = n_celltypes
        self.use_celltype = n_celltypes > 0
        self.celltype_embedding_dim = celltype_embedding_dim
        self.dropout = nn.Dropout(dropout)
        
        self.phase_expansion_factor = phase_expansion_factor
        self.adaptive_expansion = adaptive_expansion
        self.min_expansion_factor = 1.0
        self.max_expansion_factor = 3.0
        
        if self.use_celltype:
            self.celltype_embedding = nn.Embedding(n_celltypes, celltype_embedding_dim)
            self.scale_transform = nn.Linear(celltype_embedding_dim, input_dim)
            self.additive_transform = nn.Linear(celltype_embedding_dim, input_dim)
            self.global_bias = nn.Parameter(torch.zeros(input_dim))
            encoder_input_dim = input_dim
        else:
            encoder_input_dim = input_dim
        
        self.encoder = nn.Linear(encoder_input_dim, 2)
        self.decoder = nn.Linear(2, input_dim)
    
    def progressive_phase_expansion_by_celltype(self, phase_coords_normalized, celltype_indices):
        if celltype_indices is None:
            return self.progressive_phase_expansion(phase_coords_normalized)
        
        expanded_coords = torch.zeros_like(phase_coords_normalized)
        unique_celltypes = torch.unique(celltype_indices)
        
        for celltype in unique_celltypes:
            print(celltype)
            celltype_mask = (celltype_indices == celltype)
            celltype_coords = phase_coords_normalized[celltype_mask]
            batch_size = celltype_coords.shape[0]
            
            if batch_size <= 2:
                expanded_coords[celltype_mask] = celltype_coords
                continue
            
            angles = torch.atan2(celltype_coords[:, 1], celltype_coords[:, 0])
            print("angle: ", angles)
            angle_mean = torch.atan2(torch.mean(celltype_coords[:, 1]), 
                                   torch.mean(celltype_coords[:, 0]))
            
            angle_deviations = angles - angle_mean
            
            angle_deviations = torch.atan2(torch.sin(angle_deviations), 
                                         torch.cos(angle_deviations))
            
            current_range = torch.max(angle_deviations) - torch.min(angle_deviations)
            
            target_range = min(2 * np.pi * 0.9, current_range * self.phase_expansion_factor)

            expansion_ratio = target_range / current_range
            
            expanded_deviations = angle_deviations * expansion_ratio
            expanded_angles = angle_mean + expanded_deviations
            print("expanded_angles: ", expanded_angles)
            expanded_celltype_coords = torch.stack([
                torch.cos(expanded_angles),
                torch.sin(expanded_angles)
            ], dim=1)
            
            expanded_coords[celltype_mask] = expanded_celltype_coords
        
        return expanded_coords
    
    def forward(self, x, celltype_indices=None, expansion_mode='progressive'):
        if self.use_celltype and celltype_indices is not None:
            celltype_emb = self.celltype_embedding(celltype_indices)
            scale_factor = self.scale_transform(celltype_emb)
            additive_factor = self.additive_transform(celltype_emb)
            modified_input = x * (1 + scale_factor) + additive_factor + self.global_bias
        else:
            modified_input = x
        
        phase_coords = self.encoder(modified_input)
        norm = torch.norm(phase_coords, dim=1, keepdim=True) + 1e-8
        phase_coords_normalized = phase_coords / norm
        
        if self.training:
            expanded_phase_coords = phase_coords_normalized
        else:
            if expansion_mode == 'progressive':
                expanded_phase_coords = self.progressive_phase_expansion_by_celltype(
                    phase_coords_normalized, celltype_indices)
            else:
                expanded_phase_coords = phase_coords_normalized
        
        reconstructed = self.decoder(expanded_phase_coords)
        
        return expanded_phase_coords, reconstructed
    
    def encode(self, x, celltype_indices=None, expansion_mode='progressive'):
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
        
        if expansion_mode == 'progressive':
            expanded_phase_coords = self.progressive_phase_expansion_by_celltype(
                phase_coords_normalized, celltype_indices)
        else:
            expanded_phase_coords = phase_coords_normalized
        
        return expanded_phase_coords
    
    def decode(self, phase_coords):
        return self.decoder(phase_coords)