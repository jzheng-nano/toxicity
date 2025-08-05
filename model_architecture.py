import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATConv, global_mean_pool, BatchNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CrystalToxicityTransformer(nn.Module):
    """
    Enhanced crystal toxicity predictor with GNN and Transformer
    Combines graph neural networks for crystal structure with Transformer for experimental features
    """
    def __init__(self, exp_feature_dim, gnn_hidden_dim=256, transformer_dim=256, 
                 nhead=8, n_layers=3, dropout_rate=0.2):
        super().__init__()
        self.gnn_hidden_dim = gnn_hidden_dim
        self.transformer_dim = transformer_dim
        self.dropout = nn.Dropout(dropout_rate)
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, gnn_hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(gnn_hidden_dim),
            self.dropout
        )
        
        # Graph Neural Network Encoder
        self.gnn_layers = nn.ModuleList()
        self.gnn_batch_norms = nn.ModuleList()
        
        # First GNN layer
        self.gnn_layers.append(
            GINEConv(
                nn.Sequential(
                    nn.Linear(16, gnn_hidden_dim),
                    nn.LeakyReLU(0.1),
                    nn.LayerNorm(gnn_hidden_dim),
                    self.dropout,
                    nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
                ),
                edge_dim=gnn_hidden_dim
            )
        )
        self.gnn_batch_norms.append(BatchNorm(gnn_hidden_dim))
        
        # Additional GNN layers
        for _ in range(2):
            self.gnn_layers.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
                        nn.LeakyReLU(0.1),
                        nn.LayerNorm(gnn_hidden_dim),
                        self.dropout,
                        nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
                    ),
                    edge_dim=gnn_hidden_dim
                )
            )
            self.gnn_batch_norms.append(BatchNorm(gnn_hidden_dim))
        
        # Graph Attention Layer
        self.gat_conv = GATConv(gnn_hidden_dim, gnn_hidden_dim // 4, heads=4, dropout=dropout_rate)
        self.gat_norm = BatchNorm(gnn_hidden_dim)
        
        # Transformer Encoder for Experimental Features
        self.exp_projection = nn.Sequential(
            nn.Linear(exp_feature_dim, transformer_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(transformer_dim),
            self.dropout
        )
        
        # Positional encoding (learnable)
        self.pos_encoder = nn.Embedding(64, transformer_dim)  # Max 64 features
        
        # Transformer layers
        transformer_layer = TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=4*transformer_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=n_layers)
        
        # Feature Fusion and Prediction
        self.fusion_layer = nn.Sequential(
            nn.Linear(gnn_hidden_dim + transformer_dim, 2*transformer_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(2*transformer_dim),
            self.dropout
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(2*transformer_dim, transformer_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(transformer_dim),
            self.dropout,
            nn.Linear(transformer_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for different layer types"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward_gnn(self, graph_data):
        """Process crystal structure data through GNN layers"""
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        
        # Handle empty graphs or single-atom cases
        if edge_index.size(1) == 0:
            num_nodes = x.size(0)
            edge_index = torch.tensor([[i, i] for i in range(num_nodes)], 
                                     dtype=torch.long, device=x.device).t().contiguous()
            edge_attr = torch.ones((num_nodes, 1), device=x.device) * 0.1
        
        # Encode edge features
        edge_embed = self.edge_encoder(edge_attr)
        
        # Process through GNN layers
        for i, (conv, bn) in enumerate(zip(self.gnn_layers, self.gnn_batch_norms)):
            x = conv(x, edge_index, edge_embed)
            x = F.leaky_relu(bn(x), 0.1)
            x = self.dropout(x)
        
        # Apply graph attention
        x = self.gat_conv(x, edge_index)
        x = F.leaky_relu(self.gat_norm(x), 0.1)
        x = self.dropout(x)
        
        # Global pooling to get graph representation
        return global_mean_pool(x, graph_data.batch)
    
    def forward(self, graph_data, exp_features):
        """
        Forward pass:
        - graph_data: Crystal structure data (PyG Data object)
        - exp_features: Experimental features tensor [batch_size, exp_feature_dim]
        """
        # Process crystal structure
        graph_embed = self.forward_gnn(graph_data)  # [batch_size, gnn_hidden_dim]
        
        # Process experimental features with Transformer
        batch_size = exp_features.size(0)
        exp_embed = self.exp_projection(exp_features)  # [batch_size, transformer_dim]
        
        # Create sequence input with positional encoding
        seq_length = 1  # We treat each sample as a sequence of length 1
        positions = torch.arange(seq_length, device=exp_embed.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embed = self.pos_encoder(positions)
        
        # Combine feature and positional embeddings
        transformer_input = exp_embed.unsqueeze(1) + pos_embed  # [batch_size, seq_len, transformer_dim]
        
        # Process through Transformer
        transformer_output = self.transformer_encoder(transformer_input)
        exp_embed = transformer_output.squeeze(1)  # [batch_size, transformer_dim]
        
        # Feature fusion
        combined = torch.cat([graph_embed, exp_embed], dim=1)  # [batch_size, gnn_hidden_dim + transformer_dim]
        fused = self.fusion_layer(combined)
        
        # Final prediction
        return self.predictor(fused).squeeze(-1)  # [batch_size]