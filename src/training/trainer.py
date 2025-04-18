"""
Trainer module for dimensional cascade models.

This module provides a Trainer class for training dimension distillation
models with support for various callbacks and customizations.
"""

import os
import time
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pathlib import Path

# Import our callbacks
from src.training.callbacks import (
    Callback, CallbackList, History, EarlyStopping, 
    ModelCheckpoint, LearningRateScheduler, TensorboardLogger
)

# Set up logging
logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for dimensional cascade models.
    
    This class handles the training process for dimension distillation models,
    including validation, evaluation, and callback support.
    
    Attributes:
        model: The PyTorch model to train
        optimizer: The optimizer to use for training
        loss_fn: The loss function to use for training
        device: The device to train on (CPU or GPU)
        callbacks: List of callbacks to use during training
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        """Initialize the trainer.
        
        Args:
            model: The PyTorch model to train
            optimizer: The optimizer to use for training
            loss_fn: The loss function to use for training
            device: The device to train on (CPU or GPU), defaults to CUDA if available
            callbacks: List of callbacks to use during training
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Set up callbacks
        self.history = History()
        self.callbacks = CallbackList([self.history])
        if callbacks:
            for callback in callbacks:
                self.callbacks.append(callback)
    
    def train_step(
        self, 
        batch: Tuple[torch.Tensor, ...], 
        batch_idx: int
    ) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            batch: Tuple of input tensors (inputs, targets)
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary of metrics for the current step
        """
        # Unpack batch
        inputs, targets = batch
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Calculate loss
        loss = self.loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def validation_step(
        self, 
        batch: Tuple[torch.Tensor, ...], 
        batch_idx: int
    ) -> Dict[str, float]:
        """Perform a single validation step.
        
        Args:
            batch: Tuple of input tensors (inputs, targets)
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary of metrics for the current step
        """
        # Unpack batch
        inputs, targets = batch
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.loss_fn(outputs, targets)
        
        return {'val_loss': loss.item()}
    
    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
    ) -> History:
        """Train the model.
        
        Args:
            train_loader: DataLoader for training data
            epochs: Number of epochs to train for
            val_loader: Optional DataLoader for validation data
            verbose: Whether to print progress during training
            
        Returns:
            History object containing training metrics
        """
        # Prepare model for training
        self.model.train()
        
        # Initialize logs dictionary
        logs = {'model': self.model, 'optimizer': self.optimizer}
        
        # Call on_train_begin for all callbacks
        self.callbacks.on_train_begin(logs)
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_logs = {'model': self.model, 'optimizer': self.optimizer}
            
            # Call on_epoch_begin for all callbacks
            self.callbacks.on_epoch_begin(epoch, epoch_logs)
            
            # Initialize batch metrics
            train_metrics = {}
            
            # Training phase
            self.model.train()
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                batch = [b.to(self.device) for b in batch]
                
                # Call on_batch_begin for all callbacks
                batch_logs = {'batch': batch_idx, 'size': len(batch[0])}
                self.callbacks.on_batch_begin(batch_idx, batch_logs)
                
                # Perform training step
                step_metrics = self.train_step(batch, batch_idx)
                
                # Update batch metrics
                batch_logs.update(step_metrics)
                
                # Call on_batch_end for all callbacks
                self.callbacks.on_batch_end(batch_idx, batch_logs)
                
                # Update epoch metrics (running average)
                for metric, value in step_metrics.items():
                    if metric not in train_metrics:
                        train_metrics[metric] = []
                    train_metrics[metric].append(value)
            
            # Compute average metrics for the epoch
            for metric, values in train_metrics.items():
                epoch_logs[metric] = np.mean(values)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_metrics = {}
                
                for batch_idx, batch in enumerate(val_loader):
                    # Move batch to device
                    batch = [b.to(self.device) for b in batch]
                    
                    # Perform validation step
                    step_metrics = self.validation_step(batch, batch_idx)
                    
                    # Update validation metrics
                    for metric, value in step_metrics.items():
                        if metric not in val_metrics:
                            val_metrics[metric] = []
                        val_metrics[metric].append(value)
                
                # Compute average metrics for the validation epoch
                for metric, values in val_metrics.items():
                    epoch_logs[metric] = np.mean(values)
            
            # Add epoch time to logs
            epoch_time = time.time() - epoch_start_time
            epoch_logs['epoch_time'] = epoch_time
            
            # Print progress
            if verbose:
                status = f"Epoch {epoch+1}/{epochs}"
                metrics_str = ' - '.join([f"{k}: {v:.4f}" for k, v in epoch_logs.items() 
                                         if k not in ['model', 'optimizer']])
                logger.info(f"{status} - {metrics_str}")
            
            # Call on_epoch_end for all callbacks
            self.callbacks.on_epoch_end(epoch, epoch_logs)
            
            # Check if training should stop
            if hasattr(self.callbacks, 'stop_training') and self.callbacks.stop_training:
                logger.info("Early stopping triggered, ending training")
                break
        
        # Call on_train_end for all callbacks
        self.callbacks.on_train_end(logs)
        
        return self.history
    
    def evaluate(
        self,
        test_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data
            verbose: Whether to print progress during evaluation
            
        Returns:
            Dictionary of test metrics
        """
        # Prepare model for evaluation
        self.model.eval()
        
        # Initialize test metrics
        test_metrics = {}
        
        # Evaluation loop
        for batch_idx, batch in enumerate(test_loader):
            # Move batch to device
            batch = [b.to(self.device) for b in batch]
            
            # Perform validation step
            step_metrics = self.validation_step(batch, batch_idx)
            
            # Update test metrics
            for metric, value in step_metrics.items():
                metric_name = metric.replace('val_', 'test_')
                if metric_name not in test_metrics:
                    test_metrics[metric_name] = []
                test_metrics[metric_name].append(value)
        
        # Compute average metrics
        avg_metrics = {}
        for metric, values in test_metrics.items():
            avg_metrics[metric] = np.mean(values)
        
        # Print results
        if verbose:
            metrics_str = ' - '.join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
            logger.info(f"Evaluation results: {metrics_str}")
        
        return avg_metrics
    
    def predict(
        self,
        data_loader: DataLoader,
        return_targets: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate predictions for the given data.
        
        Args:
            data_loader: DataLoader for the data to predict
            return_targets: Whether to also return the targets
            
        Returns:
            Predictions tensor, or tuple of (predictions, targets) if return_targets is True
        """
        # Prepare model for evaluation
        self.model.eval()
        
        # Initialize lists for predictions and targets
        all_preds = []
        all_targets = [] if return_targets else None
        
        # Prediction loop
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    if return_targets and len(batch) > 1:
                        targets = batch[1].to(self.device)
                        all_targets.append(targets)
                else:
                    inputs = batch.to(self.device)
                
                # Generate predictions
                outputs = self.model(inputs)
                all_preds.append(outputs)
        
        # Concatenate predictions
        predictions = torch.cat(all_preds, dim=0)
        
        # Return predictions and targets if requested
        if return_targets and all_targets:
            targets = torch.cat(all_targets, dim=0)
            return predictions, targets
        
        return predictions
    
    def save_model(
        self,
        filepath: str,
        save_optimizer: bool = False,
    ) -> None:
        """Save the model and optionally the optimizer.
        
        Args:
            filepath: Path to save the model file
            save_optimizer: Whether to also save the optimizer state
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare save dict
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': getattr(self.model, 'config', None),
        }
        
        if save_optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # Save the model
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(
        self,
        filepath: str,
        load_optimizer: bool = False,
    ) -> None:
        """Load the model and optionally the optimizer.
        
        Args:
            filepath: Path to the model file
            load_optimizer: Whether to also load the optimizer state
        """
        # Load the save dict
        save_dict = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(save_dict['model_state_dict'])
        
        # Load optimizer state if requested
        if load_optimizer and 'optimizer_state_dict' in save_dict:
            self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")

def create_dataloaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int = 32,
    val_split: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 0,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create DataLoaders for training and validation.
    
    Args:
        X_train: Training inputs
        y_train: Training targets
        batch_size: Batch size
        val_split: Fraction of training data to use for validation
        shuffle: Whether to shuffle the training data
        num_workers: Number of workers for DataLoader
        X_val: Optional validation inputs (if provided, val_split is ignored)
        y_val: Optional validation targets (if provided, val_split is ignored)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create validation set if not provided
    if X_val is None or y_val is None:
        if val_split > 0:
            # Calculate split indices
            val_size = int(len(X_train) * val_split)
            indices = torch.randperm(len(X_train))
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            # Split the data
            X_train_split = X_train[train_indices]
            y_train_split = y_train[train_indices]
            X_val = X_train[val_indices]
            y_val = y_train[val_indices]
        else:
            # No validation
            X_train_split = X_train
            y_train_split = y_train
            X_val = None
            y_val = None
    else:
        # Use provided validation data
        X_train_split = X_train
        y_train_split = y_train
    
    # Create datasets
    train_dataset = TensorDataset(X_train_split, y_train_split)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    
    # Create validation loader if validation data is available
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
    return train_loader, val_loader


def create_default_callbacks(
    checkpoint_dir: str,
    log_dir: str,
    monitor: str = 'val_loss',
    patience: int = 10,
    save_best_only: bool = True,
    tensorboard: bool = True,
) -> List[Callback]:
    """Create a set of default callbacks.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        monitor: Metric to monitor for early stopping and checkpointing
        patience: Number of epochs with no improvement after which training will be stopped
        save_best_only: Whether to save only the best model
        tensorboard: Whether to use TensorBoard logging
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=True,
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_{epoch:03d}.pt')
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        verbose=True,
        save_best_only=save_best_only,
        mode='min' if 'loss' in monitor else 'max',
    )
    callbacks.append(checkpoint)
    
    # TensorBoard logging
    if tensorboard:
        try:
            tensorboard_logger = TensorboardLogger(log_dir=log_dir)
            callbacks.append(tensorboard_logger)
        except ImportError:
            logger.warning("TensorBoard not available, skipping TensorBoard logging")
    
    return callbacks


if __name__ == '__main__':
    # Example usage
    import torch.nn as nn
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 25),
    )
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Create synthetic data
    X_train = torch.randn(1000, 100)
    y_train = torch.randn(1000, 25)
    X_val = torch.randn(200, 100)
    y_val = torch.randn(200, 25)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, 
        batch_size=32, 
        X_val=X_val, 
        y_val=y_val
    )
    
    # Create callbacks
    callbacks = create_default_callbacks(
        checkpoint_dir='checkpoints',
        log_dir='logs',
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        callbacks=callbacks,
    )
    
    # Train the model
    history = trainer.fit(
        train_loader=train_loader,
        epochs=10,
        val_loader=val_loader,
    )
    
    # Evaluate the model
    test_metrics = trainer.evaluate(val_loader)
    
    # Save the model
    trainer.save_model('model.pt') 