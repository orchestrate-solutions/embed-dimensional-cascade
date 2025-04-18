"""
Training callbacks for dimensional cascade training.

This module provides callback classes for model training, including:
- Early stopping to prevent overfitting
- Model checkpointing to save the best models
- Training history tracking
"""

import os
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class Callback:
    """Base callback class that all callbacks inherit from."""
    
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of training.
        
        Args:
            logs: Dictionary of logs (if any)
        """
        pass
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Called at the end of training.
        
        Args:
            logs: Dictionary of logs (if any)
        """
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of an epoch.
        
        Args:
            epoch: Integer, current epoch index
            logs: Dictionary of logs (if any)
        """
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of an epoch.
        
        Args:
            epoch: Integer, current epoch index
            logs: Dictionary of logs (if any)
        """
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of a batch.
        
        Args:
            batch: Integer, current batch index
            logs: Dictionary of logs (if any)
        """
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of a batch.
        
        Args:
            batch: Integer, current batch index
            logs: Dictionary of logs (if any)
        """
        pass

class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.
    
    Attributes:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        patience: Number of epochs with no improvement after which training will be stopped.
        mode: One of {'min', 'max'}. In 'min' mode, training will stop when the quantity
            monitored has stopped decreasing; in 'max' mode it will stop when the
            quantity monitored has stopped increasing.
        baseline: Baseline value for the monitored quantity.
        restore_best_weights: Whether to restore model weights from the epoch with the
            best value of the monitored quantity.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 0,
        mode: str = 'min',
        baseline: Optional[float] = None,
        restore_best_weights: bool = False,
        verbose: bool = False
    ) -> None:
        """Initialize EarlyStopping callback.
        
        Args:
            monitor: Quantity to be monitored.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            patience: Number of epochs with no improvement after which training will be stopped.
            mode: One of {'min', 'max'}. In 'min' mode, training will stop when the quantity
                monitored has stopped decreasing; in 'max' mode it will stop when the
                quantity monitored has stopped increasing.
            baseline: Baseline value for the monitored quantity.
            restore_best_weights: Whether to restore model weights from the epoch with the
                best value of the monitored quantity.
            verbose: Whether to print progress messages.
        """
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_weights = None
        self.stopped_epoch = 0
        self.wait = 0
        self.best = None
        self.stop_training = False
        
        if mode not in ['min', 'max']:
            logger.warning(f"EarlyStopping mode {mode} is unknown, fallback to 'min' mode.")
            mode = 'min'
        
        self.mode = mode
        self.monitor_op = np.less if mode == 'min' else np.greater
        self.min_delta = min_delta if self.monitor_op == np.greater else -min_delta
    
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """Initialize the best value at the start of training."""
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
        self.best_weights = None
        
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.mode == 'min' else -np.Inf
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Check at end of epoch whether to stop training.
        
        Args:
            epoch: Current epoch index
            logs: Dictionary containing the metrics
        """
        logs = logs or {}
        
        if self.monitor not in logs:
            logger.warning(f"Early stopping metric '{self.monitor}' not found in logs. Available: {list(logs.keys())}")
            return
        
        current = logs[self.monitor]
        if current is None:
            return
        
        if self.restore_best_weights and self.best_weights is None:
            # Save a copy of the weights to restore later
            self.best_weights = {
                name: param.clone().detach() 
                for name, param in logs.get('model', {}).state_dict().items()
            } if 'model' in logs else None
        
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                # Update the copy of best weights
                self.best_weights = {
                    name: param.clone().detach() 
                    for name, param in logs.get('model', {}).state_dict().items()
                } if 'model' in logs else None
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose:
                        logger.info('Restoring model weights from the end of the best epoch')
                    if 'model' in logs:
                        logs['model'].load_state_dict(self.best_weights)
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Handle end of training.
        
        Args:
            logs: Dictionary containing the metrics
        """
        if self.stopped_epoch > 0 and self.verbose:
            logger.info(f'Epoch {self.stopped_epoch+1}: early stopping')

class ModelCheckpoint(Callback):
    """Save the model after every epoch.
    
    Attributes:
        filepath: Path to save the model file.
        monitor: Quantity to monitor.
        verbose: Verbosity mode, 0 or 1.
        save_best_only: If True, the latest best model will not be overwritten.
        mode: One of {'min', 'max'}. In 'min' mode, the model with the lowest monitored
            metric will be saved. In 'max' mode, the model with the highest monitored
            metric will be saved.
        save_weights_only: If True, then only the model's weights will be saved.
        period: Interval (number of epochs) between checkpoints.
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        verbose: bool = False,
        save_best_only: bool = False,
        mode: str = 'min',
        save_weights_only: bool = False,
        period: int = 1,
    ) -> None:
        """Initialize ModelCheckpoint callback.
        
        Args:
            filepath: Path to save the model file.
            monitor: Quantity to monitor.
            verbose: Verbosity mode, 0 or 1.
            save_best_only: If True, the latest best model will not be overwritten.
            mode: One of {'min', 'max'}. In 'min' mode, the model with the lowest monitored
                metric will be saved. In 'max' mode, the model with the highest monitored
                metric will be saved.
            save_weights_only: If True, then only the model's weights will be saved.
            period: Interval (number of epochs) between checkpoints.
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        
        if mode not in ['min', 'max']:
            logger.warning(f"ModelCheckpoint mode {mode} is unknown, fallback to 'min' mode.")
            mode = 'min'
        
        self.mode = mode
        self.monitor_op = np.less if mode == 'min' else np.greater
        self.best = np.Inf if mode == 'min' else -np.Inf
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Save the model at the end of an epoch if conditions are met.
        
        Args:
            epoch: Current epoch index
            logs: Dictionary containing the metrics
        """
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            
            # Format filepath
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            
            if self.save_best_only:
                if self.monitor not in logs:
                    logger.warning(f"Checkpoint metric '{self.monitor}' not found in logs. Available: {list(logs.keys())}")
                    return
                
                current = logs[self.monitor]
                if current is None:
                    return
                
                if self.monitor_op(current, self.best):
                    if self.verbose:
                        logger.info(f'Epoch {epoch+1}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving model to {filepath}')
                    self.best = current
                    if 'model' in logs:
                        self._save_model(logs['model'], filepath)
                else:
                    if self.verbose:
                        logger.info(f'Epoch {epoch+1}: {self.monitor} did not improve from {self.best:.5f}')
            else:
                if self.verbose:
                    logger.info(f'Epoch {epoch+1}: saving model to {filepath}')
                if 'model' in logs:
                    self._save_model(logs['model'], filepath)
    
    def _save_model(self, model: torch.nn.Module, filepath: str) -> None:
        """Save the model.
        
        Args:
            model: PyTorch model to save
            filepath: Path to save the model file
        """
        if self.save_weights_only:
            torch.save(model.state_dict(), filepath)
        else:
            torch.save(model, filepath)

class History(Callback):
    """Callback that records events into a History object.
    
    This callback is automatically applied to every model. The History
    object gets returned by the `fit` method of models.
    """
    
    def __init__(self) -> None:
        """Initialize History callback."""
        super().__init__()
        self.history = {}
    
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of training.
        
        Args:
            logs: Dictionary of logs (if any)
        """
        self.history = {}
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of an epoch.
        
        Args:
            epoch: Current epoch index
            logs: Dictionary containing the metrics
        """
        logs = logs or {}
        
        for key, value in logs.items():
            if key != 'model':  # Don't store the model in history
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
    
    def save_history(self, filepath: str) -> None:
        """Save training history to a JSON file.
        
        Args:
            filepath: Path to save the history file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            # Convert numpy arrays and other non-serializable types to Python types
            history_serializable = {}
            for key, values in self.history.items():
                history_serializable[key] = [
                    float(v) if isinstance(v, (np.number, np.ndarray)) else v
                    for v in values
                ]
            json.dump(history_serializable, f, indent=2)

class CallbackList:
    """Container for managing a list of callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """Initialize CallbackList.
        
        Args:
            callbacks: List of callbacks to manage
        """
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        """Add a callback to the list.
        
        Args:
            callback: Callback to add
        """
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of training.
        
        Args:
            logs: Dictionary of logs (if any)
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Called at the end of training.
        
        Args:
            logs: Dictionary of logs (if any)
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of an epoch.
        
        Args:
            epoch: Current epoch index
            logs: Dictionary of logs (if any)
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of an epoch.
        
        Args:
            epoch: Current epoch index
            logs: Dictionary of logs (if any)
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
            
        # Check if training should stop
        self.stop_training = any(
            getattr(callback, 'stop_training', False) 
            for callback in self.callbacks
        )
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of a batch.
        
        Args:
            batch: Current batch index
            logs: Dictionary of logs (if any)
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of a batch.
        
        Args:
            batch: Current batch index
            logs: Dictionary of logs (if any)
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

class LearningRateScheduler(Callback):
    """Learning rate scheduler callback.
    
    This callback is used to dynamically change the learning rate
    of the optimizer during training.
    
    Attributes:
        schedule: Function that takes an epoch index and current learning rate
            as inputs and returns a new learning rate.
        verbose: If True, prints a message when learning rate is updated.
    """
    
    def __init__(
        self, 
        schedule: Callable[[int, float], float], 
        verbose: bool = False
    ):
        """Initialize the learning rate scheduler.
        
        Args:
            schedule: Function that takes an epoch index and current learning rate
                as inputs and returns a new learning rate.
            verbose: If True, prints a message when learning rate is updated.
        """
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Update learning rate at the beginning of each epoch.
        
        Args:
            epoch: Current epoch index
            logs: Dictionary containing the metrics and model
        """
        if not logs or 'optimizer' not in logs:
            return
        
        optimizer = logs['optimizer']
        lrs = []
        
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = self.schedule(epoch, old_lr)
            param_group['lr'] = new_lr
            lrs.append(new_lr)
        
        if self.verbose:
            logger.info(f'Epoch {epoch+1}: setting learning rate to {lrs}')
            
class TensorboardLogger(Callback):
    """Callback to log metrics to TensorBoard.
    
    Attributes:
        log_dir: Directory where to save the log files.
    """
    
    def __init__(self, log_dir: str):
        """Initialize the TensorBoard logger.
        
        Args:
            log_dir: Directory where to save the log files.
        """
        super().__init__()
        self.log_dir = log_dir
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self.enabled = True
        except ImportError:
            logger.warning("TensorboardX not found. TensorBoard logging is disabled.")
            self.enabled = False
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Log metrics at the end of each epoch.
        
        Args:
            epoch: Current epoch index
            logs: Dictionary containing the metrics
        """
        if not self.enabled or not logs:
            return
        
        for name, value in logs.items():
            if name in ['model', 'optimizer']:  # Skip non-metric items
                continue
                
            if isinstance(value, torch.Tensor):
                value = value.item()
                
            if isinstance(value, (int, float)):
                self.writer.add_scalar(name, value, epoch)
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Close the writer at the end of training.
        
        Args:
            logs: Dictionary containing the metrics
        """
        if self.enabled:
            self.writer.close()


# Common learning rate schedulers as helper functions

def step_decay(step_size: int, gamma: float = 0.1):
    """Create a step decay learning rate schedule.
    
    Args:
        step_size: Period of learning rate decay (in epochs)
        gamma: Multiplicative factor of learning rate decay
        
    Returns:
        Function that computes the new learning rate
    """
    def schedule(epoch: int, lr: float) -> float:
        """Return the new learning rate.
        
        Args:
            epoch: Current epoch index
            lr: Current learning rate
            
        Returns:
            New learning rate
        """
        return lr * gamma ** (epoch // step_size)
    
    return schedule

def exponential_decay(gamma: float = 0.95):
    """Create an exponential decay learning rate schedule.
    
    Args:
        gamma: Multiplicative factor of learning rate decay
        
    Returns:
        Function that computes the new learning rate
    """
    def schedule(epoch: int, lr: float) -> float:
        """Return the new learning rate.
        
        Args:
            epoch: Current epoch index
            lr: Current learning rate
            
        Returns:
            New learning rate
        """
        return lr * gamma
    
    return schedule

def cosine_annealing(T_max: int, eta_min: float = 0):
    """Create a cosine annealing learning rate schedule.
    
    Args:
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate
        
    Returns:
        Function that computes the new learning rate
    """
    def schedule(epoch: int, lr: float) -> float:
        """Return the new learning rate.
        
        Args:
            epoch: Current epoch index
            lr: Current learning rate
            
        Returns:
            New learning rate
        """
        import math
        return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
    
    return schedule

def linear_warmup(warmup_epochs: int, peak_lr: float):
    """Create a linear warmup learning rate schedule.
    
    Args:
        warmup_epochs: Number of epochs for the warmup
        peak_lr: Learning rate after warmup
        
    Returns:
        Function that computes the new learning rate
    """
    def schedule(epoch: int, lr: float) -> float:
        """Return the new learning rate.
        
        Args:
            epoch: Current epoch index
            lr: Current learning rate (unused)
            
        Returns:
            New learning rate
        """
        if epoch >= warmup_epochs:
            return peak_lr
        return peak_lr * (epoch + 1) / warmup_epochs
    
    return schedule 