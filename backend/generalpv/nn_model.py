"""
Neural Network Expected Threat Model using Attention-based architecture.
Designed for GPU training on RTX 4090.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import math

# Fix for Apple Silicon segfault - must be set BEFORE importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add project root to path
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_MODULE_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.generalpv.nn_preprocessor import (
    preprocess_dataset, 
    process_inference_input,
    PITCH_LENGTH, 
    PITCH_WIDTH
)


# ============================================================================
# DATASET
# ============================================================================

class xTDataset(Dataset):
    """PyTorch Dataset for Expected Threat data."""
    
    def __init__(self, samples: List[Dict], max_teammates: int = 12, max_opponents: int = 12):
        """
        Args:
            samples: List of preprocessed sample dicts
            max_teammates: Maximum teammates to pad to
            max_opponents: Maximum opponents to pad to
        """
        self.samples = samples
        self.max_teammates = max_teammates
        self.max_opponents = max_opponents
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Ball features: [8]
        ball_feat = torch.tensor(sample['ball_features'], dtype=torch.float32)
        
        # Teammate features: pad to [max_teammates, 10]
        tm_feats = sample['teammate_features']
        n_tm = len(tm_feats)
        if n_tm > 0:
            tm_tensor = torch.tensor(np.stack(tm_feats), dtype=torch.float32)
        else:
            tm_tensor = torch.zeros((0, 10), dtype=torch.float32)
        
        # Pad teammates
        if n_tm < self.max_teammates:
            padding = torch.zeros((self.max_teammates - n_tm, 10), dtype=torch.float32)
            tm_tensor = torch.cat([tm_tensor, padding], dim=0)
        else:
            tm_tensor = tm_tensor[:self.max_teammates]
            n_tm = self.max_teammates
        
        # Teammate mask: 1 for real, 0 for padding
        tm_mask = torch.zeros(self.max_teammates, dtype=torch.float32)
        tm_mask[:n_tm] = 1.0
        
        # Opponent features: pad to [max_opponents, 10]
        opp_feats = sample['opponent_features']
        n_opp = len(opp_feats)
        if n_opp > 0:
            opp_tensor = torch.tensor(np.stack(opp_feats), dtype=torch.float32)
        else:
            opp_tensor = torch.zeros((0, 10), dtype=torch.float32)
        
        # Pad opponents
        if n_opp < self.max_opponents:
            padding = torch.zeros((self.max_opponents - n_opp, 10), dtype=torch.float32)
            opp_tensor = torch.cat([opp_tensor, padding], dim=0)
        else:
            opp_tensor = opp_tensor[:self.max_opponents]
            n_opp = self.max_opponents
        
        # Opponent mask
        opp_mask = torch.zeros(self.max_opponents, dtype=torch.float32)
        opp_mask[:n_opp] = 1.0
        
        # Global features: [10]
        global_feat = torch.tensor(sample['global_features'], dtype=torch.float32)
        
        # Label
        label = torch.tensor(sample['label'], dtype=torch.float32)
        
        return {
            'ball_features': ball_feat,
            'teammate_features': tm_tensor,
            'teammate_mask': tm_mask,
            'opponent_features': opp_tensor,
            'opponent_mask': opp_mask,
            'global_features': global_feat,
            'label': label,
        }


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class PlayerEncoder(nn.Module):
    """Encode individual player features into embeddings."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_players, input_dim]
        Returns:
            [batch, n_players, output_dim]
        """
        return self.mlp(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for player interactions."""
    
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, n_players, embed_dim]
            mask: [batch, n_players] - 1 for real, 0 for padding
        Returns:
            [batch, n_players, embed_dim]
        """
        batch_size = x.size(0)
        
        # Handle samples with all positions masked (no real players)
        if mask is not None:
            has_players = mask.sum(dim=-1) > 0  # [batch]
            if not has_players.all():
                # Some samples have no players - process them separately
                output = torch.zeros_like(x)
                
                if has_players.any():
                    # Process valid samples
                    valid_x = x[has_players]
                    valid_mask = mask[has_players]
                    key_padding_mask = (valid_mask == 0)
                    
                    attn_out, _ = self.attention(valid_x, valid_x, valid_x, key_padding_mask=key_padding_mask)
                    attn_out = torch.nan_to_num(attn_out, nan=0.0)
                    valid_output = self.norm(valid_x + self.dropout(attn_out))
                    
                    output[has_players] = valid_output
                
                return output
        
        # Normal path: all samples have at least one player
        if mask is not None:
            key_padding_mask = (mask == 0)
        else:
            key_padding_mask = None
        
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        x = self.norm(x + self.dropout(attn_out))
        
        return x


class AttentionPooling(nn.Module):
    """Attention-weighted pooling to get fixed-size representation."""
    
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.attention_weights = nn.Linear(embed_dim, 1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, n_players, embed_dim]
            mask: [batch, n_players] - 1 for real, 0 for padding
        Returns:
            [batch, embed_dim]
        """
        batch_size, n_players, embed_dim = x.shape
        
        # Check if any players exist per sample
        if mask is not None:
            valid_counts = mask.sum(dim=-1, keepdim=True)  # [batch, 1]
            has_players = (valid_counts > 0).squeeze(-1)  # [batch]
        else:
            has_players = torch.ones(batch_size, device=x.device, dtype=torch.bool)
        
        # For samples with no players, return zeros directly (avoid unstable softmax)
        output = torch.zeros(batch_size, embed_dim, device=x.device, dtype=x.dtype)
        
        if not has_players.any():
            return output
        
        # Only process samples that have players
        valid_x = x[has_players]  # [n_valid, n_players, embed_dim]
        valid_mask = mask[has_players] if mask is not None else None
        
        # Compute attention scores
        scores = self.attention_weights(valid_x).squeeze(-1)  # [n_valid, n_players]
        
        # Mask out padding
        if valid_mask is not None:
            scores = scores.masked_fill(valid_mask == 0, -1e4)  # Use -1e4, not -1e9
        
        # Softmax to get weights
        weights = F.softmax(scores, dim=-1)  # [n_valid, n_players]
        
        # Weighted sum
        valid_output = torch.bmm(weights.unsqueeze(1), valid_x).squeeze(1)  # [n_valid, embed_dim]
        
        # Put back valid outputs
        output[has_players] = valid_output
        
        return output


class CrossAttention(nn.Module):
    """Cross-attention between two sets of players (e.g., teammates attending to opponents)."""
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
    
    def forward(
        self,
        query: torch.Tensor,  # [batch, n_query, embed_dim]
        key_value: torch.Tensor,  # [batch, n_kv, embed_dim]
        query_mask: Optional[torch.Tensor] = None,  # [batch, n_query]
        kv_mask: Optional[torch.Tensor] = None,  # [batch, n_kv]
    ) -> torch.Tensor:
        """
        Cross-attention: query attends to key_value.
        Returns: [batch, n_query, embed_dim]
        """
        batch_size = query.size(0)
        
        # Handle samples with all positions masked
        if query_mask is not None and kv_mask is not None:
            has_query = query_mask.sum(dim=-1) > 0  # [batch]
            has_kv = kv_mask.sum(dim=-1) > 0  # [batch]
            valid = has_query & has_kv
            
            if not valid.all():
                output = query.clone()  # Start with residual
                
                if valid.any():
                    valid_query = query[valid]
                    valid_kv = key_value[valid]
                    valid_kv_mask = (kv_mask[valid] == 0)  # True = ignore
                    
                    attn_out, _ = self.attention(
                        valid_query, valid_kv, valid_kv,
                        key_padding_mask=valid_kv_mask
                    )
                    attn_out = torch.nan_to_num(attn_out, nan=0.0)
                    valid_output = self.norm(valid_query + self.dropout(attn_out))
                    output[valid] = valid_output
                
                return output
        
        # Normal path
        kv_padding_mask = (kv_mask == 0) if kv_mask is not None else None
        
        attn_out, _ = self.attention(query, key_value, key_value, key_padding_mask=kv_padding_mask)
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        output = self.norm(query + self.dropout(attn_out))
        
        return output


# ============================================================================
# MAIN MODEL
# ============================================================================

class ExpectedThreatNN(nn.Module):
    """
    Attention-based Expected Threat model.
    
    Architecture:
    1. Encode players with shared MLP
    2. Self-attention within teammates and opponents
    3. Cross-attention between groups
    4. Attention pooling to fixed size
    5. Fusion MLP for final prediction
    """
    
    def __init__(
        self,
        ball_dim: int = 8,
        player_dim: int = 10,
        global_dim: int = 20,  # Expanded from 10
        embed_dim: int = 128,  # Increased from 64
        num_attention_heads: int = 8,  # Increased from 4
        num_attention_layers: int = 4,  # v4 config
        fusion_hidden_dim: int = 512,  # Increased from 256
        dropout: float = 0.1,  # v4 config
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Player encoders (shared between teammates and opponents)
        self.player_encoder = PlayerEncoder(
            input_dim=player_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim
        )
        
        # Self-attention layers for teammates
        self.teammate_attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(embed_dim, num_attention_heads, dropout)
            for _ in range(num_attention_layers)
        ])
        
        # Self-attention layers for opponents
        self.opponent_attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(embed_dim, num_attention_heads, dropout)
            for _ in range(num_attention_layers)
        ])
        
        # Cross-attention: teammates attend to opponents and vice versa
        self.tm_to_opp_cross_attention = CrossAttention(embed_dim, num_attention_heads, dropout)
        self.opp_to_tm_cross_attention = CrossAttention(embed_dim, num_attention_heads, dropout)
        
        # Attention pooling
        self.teammate_pooling = AttentionPooling(embed_dim)
        self.opponent_pooling = AttentionPooling(embed_dim)
        
        # Ball feature projection (deeper)
        self.ball_projection = nn.Sequential(
            nn.Linear(ball_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
        )
        
        # Global feature projection (deeper)
        self.global_projection = nn.Sequential(
            nn.Linear(global_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
        )
        
        # Fusion with residual blocks
        # Input: ball(64) + teammate_ctx(128) + opponent_ctx(128) + global(64) = 384
        fusion_input_dim = embed_dim // 2 + embed_dim + embed_dim + embed_dim // 2
        
        # Initial projection to fusion_hidden_dim
        self.fusion_input = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
        )
        
        # Residual blocks
        self.fusion_res1 = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
        )
        
        self.fusion_res2 = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
        )
        
        # Downsample for residual connection
        self.fusion_downsample = nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2)
        
        # Final output
        self.fusion_output = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        ball_features: torch.Tensor,
        teammate_features: torch.Tensor,
        teammate_mask: torch.Tensor,
        opponent_features: torch.Tensor,
        opponent_mask: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            ball_features: [batch, 8]
            teammate_features: [batch, max_teammates, 10]
            teammate_mask: [batch, max_teammates]
            opponent_features: [batch, max_opponents, 10]
            opponent_mask: [batch, max_opponents]
            global_features: [batch, 20]
        
        Returns:
            [batch] - predicted xT values (logits, use sigmoid for probability)
        """
        batch_size = ball_features.size(0)
        
        # Encode players
        tm_encoded = self.player_encoder(teammate_features)  # [batch, max_tm, embed]
        opp_encoded = self.player_encoder(opponent_features)  # [batch, max_opp, embed]
        
        # Self-attention for teammates
        for attn_layer in self.teammate_attention_layers:
            tm_encoded = attn_layer(tm_encoded, teammate_mask)
        
        # Self-attention for opponents
        for attn_layer in self.opponent_attention_layers:
            opp_encoded = attn_layer(opp_encoded, opponent_mask)
        
        # Cross-attention: teammates aware of opponents
        tm_encoded = self.tm_to_opp_cross_attention(
            tm_encoded, opp_encoded, teammate_mask, opponent_mask
        )
        
        # Cross-attention: opponents aware of teammates
        opp_encoded = self.opp_to_tm_cross_attention(
            opp_encoded, tm_encoded, opponent_mask, teammate_mask
        )
        
        # Pool to fixed size
        tm_context = self.teammate_pooling(tm_encoded, teammate_mask)  # [batch, embed]
        opp_context = self.opponent_pooling(opp_encoded, opponent_mask)  # [batch, embed]
        
        # Handle edge case: no teammates or opponents visible
        # Check if all masks are 0 for any sample
        tm_valid = teammate_mask.sum(dim=1, keepdim=True) > 0  # [batch, 1]
        opp_valid = opponent_mask.sum(dim=1, keepdim=True) > 0
        
        # Replace NaN/invalid with zeros
        tm_context = torch.where(tm_valid, tm_context, torch.zeros_like(tm_context))
        opp_context = torch.where(opp_valid, opp_context, torch.zeros_like(opp_context))
        
        # Project ball and global features
        ball_proj = self.ball_projection(ball_features)  # [batch, embed//2]
        global_proj = self.global_projection(global_features)  # [batch, embed//2]
        
        # Concatenate all
        fused = torch.cat([ball_proj, tm_context, opp_context, global_proj], dim=-1)
        
        # Residual fusion network
        x = self.fusion_input(fused)  # [batch, fusion_hidden]
        
        # Residual block 1
        x = x + self.fusion_res1(x)
        x = F.gelu(x)
        
        # Residual block 2 with downsample
        x_down = self.fusion_downsample(x)
        x = x_down + self.fusion_res2(x)
        
        # Final output
        output = self.fusion_output(x).squeeze(-1)  # [batch]
        
        return output


# ============================================================================
# FOCAL LOSS FOR CLASS IMBALANCE
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance with optional label smoothing.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-7, 
                 label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [batch]
            targets: Binary labels [batch]
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Clamp inputs to prevent extreme values
        inputs = torch.clamp(inputs, min=-20.0, max=20.0)
        
        p = torch.sigmoid(inputs)
        # Clamp probabilities for numerical stability
        p = torch.clamp(p, min=self.eps, max=1.0 - self.eps)
        
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        p_t = p * targets + (1 - p) * (1 - targets)
        # Clamp p_t as well
        p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


# ============================================================================
# TRAINING
# ============================================================================

class ExpectedThreatTrainer:
    """Training loop for Expected Threat NN."""
    
    def __init__(
        self,
        model: ExpectedThreatNN,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        focal_alpha: float = 0.75,  # Higher alpha = more weight on positive class
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,  # Mild label smoothing
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = FocalLoss(
            alpha=focal_alpha, 
            gamma=focal_gamma,
            label_smoothing=label_smoothing
        )
        
        self.scheduler = None  # Set during training
        
    def train_epoch(self, dataloader: DataLoader, log_interval: int = 100) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        n_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            ball_feat = batch['ball_features'].to(self.device, non_blocking=True)
            tm_feat = batch['teammate_features'].to(self.device, non_blocking=True)
            tm_mask = batch['teammate_mask'].to(self.device, non_blocking=True)
            opp_feat = batch['opponent_features'].to(self.device, non_blocking=True)
            opp_mask = batch['opponent_mask'].to(self.device, non_blocking=True)
            global_feat = batch['global_features'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(
                ball_feat, tm_feat, tm_mask, 
                opp_feat, opp_mask, global_feat
            )
            
            # Check for NaN in logits
            if torch.isnan(logits).any():
                print(f"\n  WARNING: NaN detected in logits at batch {batch_idx}, skipping batch")
                continue
            
            # Loss
            loss = self.criterion(logits, labels)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"\n  WARNING: NaN loss at batch {batch_idx}, skipping batch")
                continue
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Metrics
            total_loss += loss.item() * len(labels)
            preds_prob = torch.sigmoid(logits).detach().cpu().numpy()
            
            # Replace any remaining NaN with 0.5
            preds_prob = np.nan_to_num(preds_prob, nan=0.5)
            
            preds = (preds_prob > 0.5).astype(float)
            total_correct += (preds == labels.cpu().numpy()).sum()
            total_samples += len(labels)
            
            all_preds.extend(preds_prob)
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        # Convert to numpy and ensure no NaN
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_preds = np.nan_to_num(all_preds, nan=0.5)
        
        metrics = {
            'loss': total_loss / max(total_samples, 1),
            'accuracy': total_correct / max(total_samples, 1),
            'roc_auc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5,
            'pr_auc': average_precision_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0,
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation/test set."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            ball_feat = batch['ball_features'].to(self.device)
            tm_feat = batch['teammate_features'].to(self.device)
            tm_mask = batch['teammate_mask'].to(self.device)
            opp_feat = batch['opponent_features'].to(self.device)
            opp_mask = batch['opponent_mask'].to(self.device)
            global_feat = batch['global_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            logits = self.model(
                ball_feat, tm_feat, tm_mask,
                opp_feat, opp_mask, global_feat
            )
            
            # Skip if NaN
            if torch.isnan(logits).any():
                continue
            
            loss = self.criterion(logits, labels)
            if torch.isnan(loss):
                continue
            
            total_loss += loss.item() * len(labels)
            
            preds_prob = torch.sigmoid(logits).detach().cpu().numpy()
            preds_prob = np.nan_to_num(preds_prob, nan=0.5)
            
            preds = (preds_prob > 0.5).astype(float)
            total_correct += (preds == labels.cpu().numpy()).sum()
            total_samples += len(labels)
            
            all_preds.extend(preds_prob)
            all_labels.extend(labels.cpu().numpy())
        
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
        
        # Convert to numpy and ensure no NaN
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_preds = np.nan_to_num(all_preds, nan=0.5)
        
        metrics = {
            'loss': total_loss / max(total_samples, 1),
            'accuracy': total_correct / max(total_samples, 1),
            'roc_auc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5,
            'pr_auc': average_precision_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0,
            'brier': brier_score_loss(all_labels, all_preds) if len(all_preds) > 0 else 1.0,
            'pred_mean': np.mean(all_preds) if len(all_preds) > 0 else 0.5,
            'pred_std': np.std(all_preds) if len(all_preds) > 0 else 0.0,
        }
        
        return metrics


# ============================================================================
# DATA SPLITTING
# ============================================================================

def stratified_split_by_match(
    samples: List[Dict],
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split samples into train/valid/test by match_id with stratification.
    """
    np.random.seed(seed)
    
    # Group by match
    match_to_samples = {}
    match_to_goal_rate = {}
    
    for s in samples:
        mid = s['match_id']
        if mid not in match_to_samples:
            match_to_samples[mid] = []
        match_to_samples[mid].append(s)
    
    # Calculate goal rate per match
    for mid, samps in match_to_samples.items():
        labels = [s['label'] for s in samps]
        match_to_goal_rate[mid] = np.mean(labels)
    
    # Stratify by goal rate
    match_ids = np.array(list(match_to_samples.keys()))
    goal_rates = np.array([match_to_goal_rate[m] for m in match_ids])
    
    median_rate = np.median(goal_rates)
    high_goal_matches = match_ids[goal_rates >= median_rate]
    low_goal_matches = match_ids[goal_rates < median_rate]
    
    np.random.shuffle(high_goal_matches)
    np.random.shuffle(low_goal_matches)
    
    def split_array(arr, train_r, valid_r):
        n = len(arr)
        train_end = int(n * train_r)
        valid_end = int(n * (train_r + valid_r))
        return arr[:train_end], arr[train_end:valid_end], arr[valid_end:]
    
    high_train, high_valid, high_test = split_array(high_goal_matches, train_ratio, valid_ratio)
    low_train, low_valid, low_test = split_array(low_goal_matches, train_ratio, valid_ratio)
    
    train_matches = set(np.concatenate([high_train, low_train]))
    valid_matches = set(np.concatenate([high_valid, low_valid]))
    test_matches = set(np.concatenate([high_test, low_test]))
    
    train_samples = [s for s in samples if s['match_id'] in train_matches]
    valid_samples = [s for s in samples if s['match_id'] in valid_matches]
    test_samples = [s for s in samples if s['match_id'] in test_matches]
    
    return train_samples, valid_samples, test_samples


# ============================================================================
# INFERENCE WRAPPER
# ============================================================================

class ExpectedThreatPredictor:
    """Wrapper for inference."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = ExpectedThreatNN(**checkpoint['model_config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.max_teammates = checkpoint.get('max_teammates', 12)
        self.max_opponents = checkpoint.get('max_opponents', 12)
    
    @torch.no_grad()
    def predict(
        self,
        ball_x: float,
        ball_y: float,
        teammates: List[Dict[str, float]],
        opponents: List[Dict[str, float]],
        team_keepers: List[Dict[str, float]] = None,
        opponent_keepers: List[Dict[str, float]] = None,
    ) -> float:
        """
        Predict xT for a single position.
        
        Args:
            ball_x, ball_y: Ball position (StatsBomb coords)
            teammates: List of {'x': float, 'y': float}
            opponents: List of {'x': float, 'y': float}
            team_keepers: Optional keepers on own team
            opponent_keepers: Optional opponent keepers
        
        Returns:
            xT probability in [0, 1]
        """
        # Process input
        processed = process_inference_input(
            ball_x, ball_y, teammates, opponents,
            team_keepers, opponent_keepers
        )
        
        # Create single-item batch
        dataset = xTDataset(
            [processed], 
            max_teammates=self.max_teammates,
            max_opponents=self.max_opponents
        )
        sample = dataset[0]
        
        # Add batch dimension and move to device
        ball_feat = sample['ball_features'].unsqueeze(0).to(self.device)
        tm_feat = sample['teammate_features'].unsqueeze(0).to(self.device)
        tm_mask = sample['teammate_mask'].unsqueeze(0).to(self.device)
        opp_feat = sample['opponent_features'].unsqueeze(0).to(self.device)
        opp_mask = sample['opponent_mask'].unsqueeze(0).to(self.device)
        global_feat = sample['global_features'].unsqueeze(0).to(self.device)
        
        # Predict
        logit = self.model(
            ball_feat, tm_feat, tm_mask,
            opp_feat, opp_mask, global_feat
        )
        
        prob = torch.sigmoid(logit).item()
        return prob


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def train_model(
    data_path: str = None,
    model_save_path: str = None,
    epochs: int = 50,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    device: str = 'cuda',
):
    """
    Main training function.
    """
    # Paths
    if data_path is None:
        data_path = os.path.join(_MODULE_DIR, "statsbomb_chained_dataset.csv")
    if model_save_path is None:
        model_save_path = os.path.join(_MODULE_DIR, "..", "models", "xt_nn_model.pt")
    
    processed_cache = os.path.join(_MODULE_DIR, "nn_processed_data.pkl")
    
    print("=" * 70)
    print("       EXPECTED THREAT NEURAL NETWORK TRAINING")
    print("=" * 70)
    
    # Check device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Clear any existing cache
        torch.cuda.empty_cache()
        print(f"  CUDA Version: {torch.version.cuda}")
        print("  ✓ GPU is ready!")
    else:
        print("  WARNING: Running on CPU - this will be slow!")
    
    # Load or preprocess data
    if os.path.exists(processed_cache):
        print(f"\nLoading cached preprocessed data from {processed_cache}...")
        with open(processed_cache, 'rb') as f:
            samples = pickle.load(f)
        print(f"  Loaded {len(samples)} samples")
    else:
        print(f"\nPreprocessing data from {data_path}...")
        samples = preprocess_dataset(data_path, processed_cache)
    
    # Split data
    print("\nSplitting data (stratified by match)...")
    train_samples, valid_samples, test_samples = stratified_split_by_match(samples)
    
    train_labels = [s['label'] for s in train_samples]
    valid_labels = [s['label'] for s in valid_samples]
    test_labels = [s['label'] for s in test_samples]
    
    print(f"  Train: {len(train_samples)} samples, goal rate: {np.mean(train_labels)*100:.2f}%")
    print(f"  Valid: {len(valid_samples)} samples, goal rate: {np.mean(valid_labels)*100:.2f}%")
    print(f"  Test:  {len(test_samples)} samples, goal rate: {np.mean(test_labels)*100:.2f}%")
    
    # Create datasets and dataloaders
    max_teammates = 12
    max_opponents = 12
    
    train_dataset = xTDataset(train_samples, max_teammates, max_opponents)
    valid_dataset = xTDataset(valid_samples, max_teammates, max_opponents)
    test_dataset = xTDataset(test_samples, max_teammates, max_opponents)
    
    # num_workers=0 to avoid multiprocessing memory issues in WSL
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model (version 4 architecture - matches saved model)
    model_config = {
        'ball_dim': 8,
        'player_dim': 10,
        'global_dim': 20,  # Expanded features
        'embed_dim': 128,  # Keep capacity
        'num_attention_heads': 8,
        'num_attention_layers': 4,  # v4 uses 4 layers
        'fusion_hidden_dim': 512,
        'dropout': 0.1,  # Low dropout
    }
    
    model = ExpectedThreatNN(**model_config)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer - go back to simpler settings that worked
    trainer = ExpectedThreatTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate * 0.5,
        weight_decay=0.01,
        focal_alpha=0.75,  # Back to original
        focal_gamma=2.0,   # Back to original
        label_smoothing=0.01,  # Minimal smoothing
    )
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    
    # Use OneCycleLR for better convergence
    trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        trainer.optimizer,
        max_lr=learning_rate * 0.5,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100,
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("       TRAINING")
    print("=" * 70)
    print(f"{'Epoch':>6} | {'Train Loss':>10} {'Train AUC':>10} {'Train PR':>10} | "
          f"{'Valid Loss':>10} {'Valid AUC':>10} {'Valid PR':>10} | {'Status':>10}")
    print("-" * 100)
    
    best_valid_auc = 0.0
    best_epoch = 0
    patience = 7
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        valid_metrics = trainer.evaluate(valid_loader)
        
        # Determine status
        status = ""
        if valid_metrics['roc_auc'] > best_valid_auc:
            best_valid_auc = valid_metrics['roc_auc']
            best_epoch = epoch
            patience_counter = 0
            status = "✓ BEST"
            
            # Save checkpoint
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'model_config': model_config,
                'max_teammates': max_teammates,
                'max_opponents': max_opponents,
                'valid_metrics': valid_metrics,
            }, model_save_path)
        else:
            patience_counter += 1
            status = f"wait {patience_counter}/{patience}"
        
        # Print progress
        print(f"{epoch:>6} | {train_metrics['loss']:>10.4f} {train_metrics['roc_auc']:>10.4f} {train_metrics['pr_auc']:>10.4f} | "
              f"{valid_metrics['loss']:>10.4f} {valid_metrics['roc_auc']:>10.4f} {valid_metrics['pr_auc']:>10.4f} | {status:>10}")
        
        # Clear GPU cache periodically
        if device.type == 'cuda' and epoch % 5 == 0:
            torch.cuda.empty_cache()
        
        if patience_counter >= patience:
            print(f"\n>>> Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 70)
    print("       FINAL EVALUATION (Best Model)")
    print("=" * 70)
    
    checkpoint = torch.load(model_save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get predictions for calibration
    print("\nFitting calibration layer (isotonic regression)...")
    from sklearn.isotonic import IsotonicRegression
    
    model.eval()
    valid_preds = []
    valid_labels = []
    
    with torch.no_grad():
        for batch in valid_loader:
            ball_feat = batch['ball_features'].to(device, non_blocking=True)
            tm_feat = batch['teammate_features'].to(device, non_blocking=True)
            tm_mask = batch['teammate_mask'].to(device, non_blocking=True)
            opp_feat = batch['opponent_features'].to(device, non_blocking=True)
            opp_mask = batch['opponent_mask'].to(device, non_blocking=True)
            global_feat = batch['global_features'].to(device, non_blocking=True)
            labels = batch['label']
            
            logits = model(ball_feat, tm_feat, tm_mask, opp_feat, opp_mask, global_feat)
            probs = torch.sigmoid(logits).cpu().numpy()
            valid_preds.extend(probs)
            valid_labels.extend(labels.numpy())
    
    valid_preds = np.array(valid_preds)
    valid_labels = np.array(valid_labels)
    
    # Fit isotonic regression calibrator
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    calibrator.fit(valid_preds, valid_labels)
    
    # Evaluate on test set with calibration
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            ball_feat = batch['ball_features'].to(device, non_blocking=True)
            tm_feat = batch['teammate_features'].to(device, non_blocking=True)
            tm_mask = batch['teammate_mask'].to(device, non_blocking=True)
            opp_feat = batch['opponent_features'].to(device, non_blocking=True)
            opp_mask = batch['opponent_mask'].to(device, non_blocking=True)
            global_feat = batch['global_features'].to(device, non_blocking=True)
            labels = batch['label']
            
            logits = model(ball_feat, tm_feat, tm_mask, opp_feat, opp_mask, global_feat)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_preds.extend(probs)
            test_labels.extend(labels.numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    # Uncalibrated metrics
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    uncalib_metrics = {
        'roc_auc': roc_auc_score(test_labels, test_preds),
        'pr_auc': average_precision_score(test_labels, test_preds),
        'brier': brier_score_loss(test_labels, test_preds),
        'pred_mean': test_preds.mean(),
        'pred_std': test_preds.std(),
    }
    
    # Calibrated predictions
    test_preds_calibrated = calibrator.predict(test_preds)
    calib_metrics = {
        'roc_auc': roc_auc_score(test_labels, test_preds_calibrated),
        'pr_auc': average_precision_score(test_labels, test_preds_calibrated),
        'brier': brier_score_loss(test_labels, test_preds_calibrated),
        'pred_mean': test_preds_calibrated.mean(),
        'pred_std': test_preds_calibrated.std(),
    }
    
    print(f"\nBest model from epoch {best_epoch}:")
    print(f"\n  {'Metric':<15} {'Uncalibrated':>12} {'Calibrated':>12}")
    print(f"  {'-'*40}")
    print(f"  {'ROC-AUC':<15} {uncalib_metrics['roc_auc']:>12.4f} {calib_metrics['roc_auc']:>12.4f}")
    print(f"  {'PR-AUC':<15} {uncalib_metrics['pr_auc']:>12.4f} {calib_metrics['pr_auc']:>12.4f}")
    print(f"  {'Brier Score':<15} {uncalib_metrics['brier']:>12.4f} {calib_metrics['brier']:>12.4f}")
    print(f"  {'Pred Mean':<15} {uncalib_metrics['pred_mean']:>12.4f} {calib_metrics['pred_mean']:>12.4f}")
    print(f"  {'Pred Std':<15} {uncalib_metrics['pred_std']:>12.4f} {calib_metrics['pred_std']:>12.4f}")
    print(f"  {'Actual Mean':<15} {test_labels.mean():>12.4f} {test_labels.mean():>12.4f}")
    
    # Save calibrator with model
    import joblib
    calibrator_path = model_save_path.replace('.pt', '_calibrator.pkl')
    joblib.dump(calibrator, calibrator_path)
    print(f"\nCalibrator saved to: {calibrator_path}")
    
    # Update checkpoint with calibrator path
    checkpoint['calibrator_path'] = calibrator_path
    torch.save(checkpoint, model_save_path)
    
    print(f"Model saved to: {model_save_path}")
    print("=" * 70)
    
    return model, calib_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Expected Threat Neural Network')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (reduce if OOM)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )
