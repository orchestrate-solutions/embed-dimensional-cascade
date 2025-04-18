"""
Embedding distillation model for dimensional cascade search.

This module provides model architectures and loss functions for distilling
high-dimensional embeddings into lower-dimensional representations while
preserving semantic relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class EmbeddingDistillationModel(nn.Module):
    """
    Neural network model for embedding distillation.
    
    This model maps high-dimensional embeddings to lower-dimensional ones
    while preserving the semantic relationships between vectors.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: Optional[List[int]] = None,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        """
        Initialize the distillation model.
        
        Args:
            input_dim: Dimension of teacher embeddings
            output_dim: Target dimension for student embeddings
            hidden_dims: Dimensions of hidden layers (if None, will use a default architecture)
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate to apply between layers (0.0 means no dropout)
        """
        super().__init__()
        
        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            # Default architecture with two hidden layers
            hidden_dims = [max(input_dim // 2, output_dim * 2), max(input_dim // 4, output_dim)]
        
        # Build the layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
                
            prev_dim = hidden_dim
        
        # Final projection layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights with Xavier/Glorot initialization
        self._init_weights()
        
        logger.info(f"Created EmbeddingDistillationModel: {input_dim}d → {output_dim}d")
        
    def _init_weights(self):
        """Initialize model weights for better training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the distillation model.
        
        Args:
            x: Input tensor of embeddings with shape (batch_size, input_dim)
            
        Returns:
            Normalized output embeddings with shape (batch_size, output_dim)
        """
        # Forward pass through layers
        embeddings = self.model(x)
        
        # L2 normalize the embeddings
        return F.normalize(embeddings, p=2, dim=1)


class DistillationLoss:
    """
    Combined loss functions for embedding distillation.
    
    Combines multiple loss terms to preserve different aspects of the 
    semantic relationships in the embedding space.
    """
    
    def __init__(
        self,
        mse_weight: float = 0.0,
        similarity_weight: float = 1.0, 
        ranking_weight: float = 1.0,
        triplet_weight: float = 0.0,
        temperature: float = 0.05,
        k: int = 10,
        triplet_margin: float = 0.2
    ):
        """
        Initialize the distillation loss.
        
        Args:
            mse_weight: Weight for direct MSE loss between projected vectors
            similarity_weight: Weight for similarity preservation loss
            ranking_weight: Weight for ranking preservation loss
            triplet_weight: Weight for triplet loss term
            temperature: Temperature for similarity scaling
            k: Number of neighbors to consider for ranking preservation
            triplet_margin: Margin for triplet loss
        """
        self.mse_weight = mse_weight
        self.similarity_weight = similarity_weight
        self.ranking_weight = ranking_weight
        self.triplet_weight = triplet_weight
        self.temperature = temperature
        self.k = k
        self.triplet_margin = triplet_margin
        
        logger.info(f"Initialized DistillationLoss with weights: "
                    f"MSE={mse_weight}, Similarity={similarity_weight}, "
                    f"Ranking={ranking_weight}, Triplet={triplet_weight}")
        
    def __call__(
        self, 
        student_embeddings: torch.Tensor, 
        teacher_embeddings: torch.Tensor,
        hard_negatives: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined distillation loss.
        
        Args:
            student_embeddings: Embeddings from student model (batch_size, student_dim)
            teacher_embeddings: Embeddings from teacher model (batch_size, teacher_dim)
            hard_negatives: Optional hard negative examples for triplet loss
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        loss_components = {}
        total_loss = 0.0
        
        # MSE loss (direct embedding matching, if applicable)
        if self.mse_weight > 0 and student_embeddings.shape[1] == teacher_embeddings.shape[1]:
            mse = F.mse_loss(student_embeddings, teacher_embeddings)
            loss_components['mse'] = mse.item()
            total_loss += self.mse_weight * mse
        
        # Similarity preservation loss
        if self.similarity_weight > 0:
            sim_loss = self._similarity_preservation_loss(student_embeddings, teacher_embeddings)
            loss_components['similarity'] = sim_loss.item()
            total_loss += self.similarity_weight * sim_loss
        
        # Ranking preservation loss
        if self.ranking_weight > 0:
            rank_loss = self._ranking_preservation_loss(student_embeddings, teacher_embeddings)
            loss_components['ranking'] = rank_loss.item()
            total_loss += self.ranking_weight * rank_loss
        
        # Triplet loss (if hard negatives provided)
        if self.triplet_weight > 0 and hard_negatives is not None:
            triplet_loss = self._triplet_loss(student_embeddings, teacher_embeddings, hard_negatives)
            loss_components['triplet'] = triplet_loss.item()
            total_loss += self.triplet_weight * triplet_loss
        
        # Add total loss to components
        loss_components['total'] = total_loss.item()
        
        return total_loss, loss_components
    
    def _similarity_preservation_loss(
        self, 
        student_embeddings: torch.Tensor, 
        teacher_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity preservation loss using KL divergence.
        
        This loss ensures that the similarity distributions between
        embeddings are preserved in the student model.
        
        Args:
            student_embeddings: Embeddings from student model
            teacher_embeddings: Embeddings from teacher model
            
        Returns:
            Similarity preservation loss
        """
        # Compute similarity matrices
        student_sim = torch.matmul(student_embeddings, student_embeddings.transpose(0, 1)) / self.temperature
        teacher_sim = torch.matmul(teacher_embeddings, teacher_embeddings.transpose(0, 1)) / self.temperature
        
        # Remove diagonal elements (self-similarity)
        mask = torch.eye(student_sim.size(0), device=student_sim.device).bool()
        student_sim = student_sim.masked_fill(mask, -1e9)  # Large negative value will become near-zero after softmax
        teacher_sim = teacher_sim.masked_fill(mask, -1e9)
        
        # KL divergence between similarity distributions
        loss = F.kl_div(
            F.log_softmax(student_sim, dim=1),
            F.softmax(teacher_sim, dim=1),
            reduction='batchmean'
        )
        
        return loss
    
    def _ranking_preservation_loss(
        self, 
        student_embeddings: torch.Tensor, 
        teacher_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ranking preservation loss to maintain nearest neighbor rankings.
        
        Args:
            student_embeddings: Embeddings from student model
            teacher_embeddings: Embeddings from teacher model
            
        Returns:
            Ranking preservation loss
        """
        # Compute similarity matrices
        student_sim = torch.matmul(student_embeddings, student_embeddings.transpose(0, 1))
        teacher_sim = torch.matmul(teacher_embeddings, teacher_embeddings.transpose(0, 1))
        
        # Mask diagonal elements (self-similarity)
        mask = torch.eye(student_sim.size(0), device=student_sim.device)
        student_sim = student_sim * (1 - mask) - mask  # Set diagonal to -1 to exclude self
        teacher_sim = teacher_sim * (1 - mask) - mask
        
        # Get top-k indices for teacher
        _, teacher_topk = teacher_sim.topk(self.k, dim=1)
        
        # Calculate recall loss
        batch_size = student_sim.size(0)
        loss = torch.tensor(0.0, device=student_sim.device)
        
        for i in range(batch_size):
            # Get student similarities for this sample
            student_sims_i = student_sim[i]
            
            # Get top-k indices in teacher space
            teacher_topk_i = teacher_topk[i]
            
            # Get student similarities for the teacher's top-k
            student_sims_for_teacher_topk = student_sims_i[teacher_topk_i]
            
            # Get student similarities for non-top-k indices in teacher space
            mask = torch.ones(batch_size, device=student_sim.device).bool()
            mask[teacher_topk_i] = False
            mask[i] = False  # Exclude self
            non_topk_indices = torch.arange(batch_size, device=student_sim.device)[mask]
            
            if len(non_topk_indices) > 0:
                student_sims_for_non_topk = student_sims_i[non_topk_indices]
                
                # Compute margin-based ranking loss
                # Each top-k item should be more similar than non-top-k items by a margin
                for topk_sim in student_sims_for_teacher_topk:
                    # Pairwise margin ranking loss
                    margins = topk_sim.unsqueeze(0) - student_sims_for_non_topk + self.triplet_margin
                    # Hinge loss: max(0, margin)
                    rank_loss = torch.clamp(margins, min=0).mean()
                    loss += rank_loss
        
        # Normalize by batch size
        return loss / batch_size if batch_size > 0 else loss
    
    def _triplet_loss(
        self, 
        student_embeddings: torch.Tensor, 
        teacher_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss to maintain relative distances between positive and negative examples.
        
        Args:
            student_embeddings: Anchor and positive embeddings from student model
            teacher_embeddings: Original teacher embeddings (used to identify positives)
            negative_embeddings: Negative embeddings from student model
            
        Returns:
            Triplet loss
        """
        batch_size = student_embeddings.size(0)
        neg_size = negative_embeddings.size(0)
        
        # Reshape for batch processing (each anchor with all negatives)
        anchors = student_embeddings.unsqueeze(1).expand(-1, neg_size, -1)  # [batch_size, neg_size, dim]
        negatives = negative_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, neg_size, dim]
        
        # Use teacher embeddings to identify positives
        # Compute similarity matrix
        teacher_sim = torch.matmul(teacher_embeddings, teacher_embeddings.transpose(0, 1))
        
        # For each anchor, find the most similar example (excluding self)
        mask = torch.eye(batch_size, device=teacher_sim.device).bool()
        teacher_sim.masked_fill_(mask, -float('inf'))
        _, pos_indices = teacher_sim.topk(1, dim=1)
        
        # Get positive embeddings
        positives = torch.stack([student_embeddings[pos_idx] for pos_idx in pos_indices.squeeze()])
        positives = positives.unsqueeze(1).expand(-1, neg_size, -1)  # [batch_size, neg_size, dim]
        
        # Compute distances
        pos_distances = torch.sum((anchors - positives) ** 2, dim=2)  # [batch_size, neg_size]
        neg_distances = torch.sum((anchors - negatives) ** 2, dim=2)  # [batch_size, neg_size]
        
        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        triplet_loss = torch.clamp(pos_distances - neg_distances + self.triplet_margin, min=0)
        
        return triplet_loss.mean()


def create_distillation_model(
    input_dim: int, 
    output_dim: int,
    hidden_dims: Optional[List[int]] = None,
    use_batch_norm: bool = True,
    dropout_rate: float = 0.0
) -> EmbeddingDistillationModel:
    """
    Factory function to create a distillation model instance.
    
    Args:
        input_dim: Dimension of teacher embeddings
        output_dim: Target dimension for student embeddings
        hidden_dims: Dimensions of hidden layers (if None, will use a default architecture)
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate to apply between layers
        
    Returns:
        Initialized EmbeddingDistillationModel
    """
    return EmbeddingDistillationModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate
    )


def create_distillation_loss(
    mse_weight: float = 0.0,
    similarity_weight: float = 1.0,
    ranking_weight: float = 1.0,
    triplet_weight: float = 0.0,
    temperature: float = 0.05,
    k: int = 10,
    triplet_margin: float = 0.2
) -> DistillationLoss:
    """
    Factory function to create a distillation loss instance.
    
    Args:
        mse_weight: Weight for direct MSE loss between projected vectors
        similarity_weight: Weight for similarity preservation loss
        ranking_weight: Weight for ranking preservation loss
        triplet_weight: Weight for triplet loss term
        temperature: Temperature for similarity scaling
        k: Number of neighbors to consider for ranking preservation
        triplet_margin: Margin for triplet loss
        
    Returns:
        Initialized DistillationLoss
    """
    return DistillationLoss(
        mse_weight=mse_weight,
        similarity_weight=similarity_weight,
        ranking_weight=ranking_weight,
        triplet_weight=triplet_weight,
        temperature=temperature,
        k=k,
        triplet_margin=triplet_margin
    ) 