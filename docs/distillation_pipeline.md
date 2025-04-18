# Dimensional Cascade: Distillation Pipeline

## Overview

The dimensional distillation pipeline is a critical component of the Dimensional Cascade system, enabling efficient semantic search across multiple vector dimensions. This document details the methodology, architecture, and implementation for creating an effective distillation pipeline that produces compact, high-quality vector embeddings while preserving semantic relationships.

## Core Concepts

Dimensional distillation leverages knowledge distillation techniques to transfer information from high-dimensional embedding models (teachers) to more compact representations (students). The key principle is to preserve semantic similarity relationships while reducing computational overhead.

Benefits of dimensional distillation:
- Reduced storage requirements
- Faster similarity computations
- Lower memory usage during inference
- Enabling efficient cascade search

## Architecture

The distillation pipeline follows a teacher-student model approach:

1. **Teacher Model**: A pre-trained embedding model producing high-dimensional vectors (e.g., 768, 1024 dims)
2. **Student Models**: Smaller networks trained to produce lower-dimensional vectors (e.g., 32, 64, 128, 256 dims)
3. **Loss Functions**: Specialized objectives that preserve semantic relationships

![Distillation Architecture](../assets/distillation_architecture.png)

## Implementation

### 1. Training Data Preparation

```python
def prepare_distillation_data(corpus_file, teacher_model, batch_size=64):
    """
    Prepare training data for distillation by generating embeddings from teacher model.
    
    Args:
        corpus_file: Path to text corpus for training
        teacher_model: Pre-trained embedding model
        batch_size: Batch size for processing
        
    Returns:
        List of normalized embeddings from teacher model
    """
    # Load corpus documents
    with open(corpus_file, 'r') as f:
        documents = [line.strip() for line in f]
    
    # Generate embeddings using teacher model
    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_embeddings = teacher_model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
    
    # Concatenate and normalize embeddings
    all_embeddings = torch.cat(embeddings, dim=0)
    normalized_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
    
    return normalized_embeddings
```

### 2. Student Model Architecture

```python
class EmbeddingDistillationModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        """
        Student model for embedding distillation.
        
        Args:
            input_dim: Dimension of teacher embeddings
            output_dim: Dimension of student embeddings
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        if hidden_dims is None:
            # Default architecture with one hidden layer
            hidden_dims = [input_dim // 2]
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass with L2 normalization of outputs"""
        embeddings = self.model(x)
        return F.normalize(embeddings, p=2, dim=1)
```

### 3. Loss Functions

#### MSE Loss
For direct embedding matching:

```python
def mse_loss(student_embeddings, teacher_embeddings):
    """Mean squared error between embeddings"""
    return F.mse_loss(student_embeddings, teacher_embeddings)
```

#### Similarity Preservation Loss
For preserving similarity relationships:

```python
def similarity_preservation_loss(student_embeddings, teacher_embeddings, temperature=0.05):
    """
    Loss that preserves similarity relationships between embeddings.
    
    Args:
        student_embeddings: Embeddings from student model
        teacher_embeddings: Embeddings from teacher model
        temperature: Temperature parameter for scaling similarities
    """
    # Compute cosine similarity matrices
    student_sim = torch.matmul(student_embeddings, student_embeddings.transpose(0, 1)) / temperature
    teacher_sim = torch.matmul(teacher_embeddings, teacher_embeddings.transpose(0, 1)) / temperature
    
    # Remove diagonal elements (self-similarity)
    mask = torch.eye(student_sim.size(0), device=student_sim.device).bool()
    student_sim = student_sim.masked_fill(mask, 0)
    teacher_sim = teacher_sim.masked_fill(mask, 0)
    
    # KL divergence loss on similarities
    loss = F.kl_div(
        F.log_softmax(student_sim, dim=1),
        F.softmax(teacher_sim, dim=1),
        reduction='batchmean'
    )
    
    return loss
```

#### Ranking Preservation Loss
For preserving nearest neighbor rankings:

```python
def ranking_preservation_loss(student_embeddings, teacher_embeddings, k=10):
    """
    Loss that preserves top-k nearest neighbor rankings.
    
    Args:
        student_embeddings: Embeddings from student model
        teacher_embeddings: Embeddings from teacher model
        k: Number of nearest neighbors to consider
    """
    # Compute cosine similarity matrices
    student_sim = torch.matmul(student_embeddings, student_embeddings.transpose(0, 1))
    teacher_sim = torch.matmul(teacher_embeddings, teacher_embeddings.transpose(0, 1))
    
    # Mask diagonal elements
    mask = torch.eye(student_sim.size(0), device=student_sim.device)
    student_sim = student_sim * (1 - mask) - mask
    teacher_sim = teacher_sim * (1 - mask) - mask
    
    # Get top-k indices for teacher
    _, teacher_topk_indices = teacher_sim.topk(k, dim=1)
    
    # Calculate precision at k
    loss = 0
    for i in range(student_sim.size(0)):
        # Get student top-k
        _, student_topk_indices = student_sim[i].topk(k)
        
        # Convert indices to sets
        student_topk_set = set(student_topk_indices.cpu().numpy())
        teacher_topk_set = set(teacher_topk_indices[i].cpu().numpy())
        
        # Calculate precision (intersection over k)
        precision = len(student_topk_set.intersection(teacher_topk_set)) / k
        loss += (1 - precision)
    
    return loss / student_sim.size(0)
```

### 4. Training Loop

```python
def train_distillation_model(teacher_embeddings, output_dim, epochs=100, batch_size=128, 
                             lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Train a distillation model to compress embeddings.
    
    Args:
        teacher_embeddings: Tensor of normalized teacher embeddings
        output_dim: Target dimension for student model
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Trained student model
    """
    input_dim = teacher_embeddings.shape[1]
    dataset = TensorDataset(teacher_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize student model
    student = EmbeddingDistillationModel(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        student.train()
        
        for batch in dataloader:
            teacher_batch = batch[0].to(device)
            
            # Forward pass
            student_batch = student(teacher_batch)
            
            # Compute loss (combination of losses)
            loss = 0.5 * similarity_preservation_loss(student_batch, teacher_batch) + \
                   0.5 * ranking_preservation_loss(student_batch, teacher_batch, k=10)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Update learning rate
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
    return student
```

### 5. Evaluation

```python
def evaluate_distilled_model(student_model, teacher_embeddings, test_embeddings=None, k=10):
    """
    Evaluate the quality of distilled embeddings.
    
    Args:
        student_model: Trained student model
        teacher_embeddings: Teacher embeddings used for evaluation
        test_embeddings: Optional separate test set
        k: Number of nearest neighbors to evaluate
        
    Returns:
        Dictionary of evaluation metrics
    """
    device = next(student_model.parameters()).device
    student_model.eval()
    
    # Use teacher embeddings if no test set provided
    if test_embeddings is None:
        test_embeddings = teacher_embeddings
    
    # Convert to device if needed
    test_embeddings = test_embeddings.to(device)
    
    # Generate student embeddings
    with torch.no_grad():
        student_embeddings = student_model(test_embeddings)
    
    # Compute similarity matrices
    teacher_sim = torch.matmul(test_embeddings, test_embeddings.transpose(0, 1))
    student_sim = torch.matmul(student_embeddings, student_embeddings.transpose(0, 1))
    
    # Mask diagonal elements
    mask = torch.eye(teacher_sim.size(0), device=device)
    teacher_sim = teacher_sim * (1 - mask) - mask
    student_sim = student_sim * (1 - mask) - mask
    
    # Get top-k indices
    _, teacher_topk = teacher_sim.topk(k, dim=1)
    _, student_topk = student_sim.topk(k, dim=1)
    
    # Calculate recall@k
    recall_sum = 0
    for i in range(teacher_sim.size(0)):
        teacher_indices = set(teacher_topk[i].cpu().numpy())
        student_indices = set(student_topk[i].cpu().numpy())
        recall = len(teacher_indices.intersection(student_indices)) / k
        recall_sum += recall
    
    avg_recall = recall_sum / teacher_sim.size(0)
    
    # Calculate Spearman rank correlation
    rank_correlation_sum = 0
    for i in range(min(100, teacher_sim.size(0))):  # Limit to first 100 for efficiency
        teacher_ranks = torch.argsort(torch.argsort(teacher_sim[i]))
        student_ranks = torch.argsort(torch.argsort(student_sim[i]))
        correlation = 1 - (6 * torch.sum((teacher_ranks - student_ranks)**2)) / (teacher_ranks.size(0) * (teacher_ranks.size(0)**2 - 1))
        rank_correlation_sum += correlation
    
    avg_rank_correlation = rank_correlation_sum / min(100, teacher_sim.size(0))
    
    return {
        "recall_at_k": avg_recall.item(),
        "rank_correlation": avg_rank_correlation.item()
    }
```

## Dimension Selection

For optimal performance, we recommend training models at the following dimensions:
- 32 dimensions: Fastest screening layer
- 64 dimensions: Improved recall over 32d, still very fast
- 128 dimensions: Good balance of quality and speed
- 256 dimensions: High-quality for final ranking

The exact dimensions should be customized based on your specific requirements and dataset characteristics.

## Integration with Cascade Search

The distilled models integrate directly into the cascade search pipeline:

1. Each dimension becomes a filter layer in the cascade
2. Lower dimensions (32d, 64d) provide fast initial screening
3. Higher dimensions (128d, 256d) refine results for better accuracy

```python
def cascade_search(query, corpus_embeddings, distilled_models, k=10):
    """
    Perform cascade search using distilled embeddings.
    
    Args:
        query: Raw query text
        corpus_embeddings: Original corpus embeddings
        distilled_models: Dictionary of dimension -> model mappings
        k: Number of results to return
    
    Returns:
        Top-k most similar document indices
    """
    # Get original query embedding
    query_embedding = get_query_embedding(query)
    
    # Initialize candidate set (all documents)
    candidates = list(range(len(corpus_embeddings['original'])))
    remaining = len(candidates)
    
    # Cascade through dimensions
    dimensions = sorted(distilled_models.keys())
    
    for dim in dimensions:
        # Skip if too few candidates remain
        if remaining <= k*2:
            break
            
        # Get distilled embeddings for this dimension
        model = distilled_models[dim]
        query_dim = model(query_embedding.unsqueeze(0)).squeeze(0)
        
        # Compute similarities for current candidates
        similarities = []
        for idx in candidates:
            doc_embedding = corpus_embeddings[dim][idx]
            similarity = torch.dot(query_dim, doc_embedding)
            similarities.append((idx, similarity))
        
        # Sort by similarity and keep top candidates
        similarities.sort(key=lambda x: x[1], reverse=True)
        keep_count = max(k * 2, remaining // 2)  # Either halve or ensure we have at least 2k
        candidates = [idx for idx, _ in similarities[:keep_count]]
        remaining = len(candidates)
    
    # Final ranking with original embeddings
    final_similarities = []
    for idx in candidates:
        doc_embedding = corpus_embeddings['original'][idx]
        similarity = torch.dot(query_embedding, doc_embedding)
        final_similarities.append((idx, similarity))
    
    final_similarities.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in final_similarities[:k]]
```

## Best Practices

1. **Normalization**: Always normalize embeddings to unit length
2. **Mixed Precision Training**: Use mixed precision for faster training
3. **Larger Batch Sizes**: Use the largest batch size your GPU memory can support
4. **Hard Negative Mining**: Incorporate challenging examples to improve discrimination
5. **Temperature Scaling**: Adjust temperature in loss functions to control sensitivity
6. **Progressive Distillation**: Consider distilling in stages (e.g., 768→256→128→64→32)

## Future Work

1. **Direct Text Distillation**: Train small models directly from text, bypassing the teacher
2. **Task-Specific Distillation**: Fine-tune for specific domains or tasks
3. **Quantization**: Apply post-training quantization for further size reduction
4. **Embedding Compression**: Explore other dimensionality reduction techniques
5. **Adaptive Cascades**: Develop dynamic cascade paths based on query characteristics

## References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. EMNLP.
3. Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., & Grave, E. (2021). Unsupervised pretraining for sentence embeddings. arXiv preprint arXiv:2104.06979. 