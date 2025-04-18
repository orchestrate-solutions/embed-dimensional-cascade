#!/usr/bin/env python
"""
Training progress monitor for dimensional cascade training.
This script parses log output from train_on_common_corpus.py and displays 
a visual representation of the training progress.

Usage:
    python track_training.py [logfile]

If logfile is not provided, it will look for the latest .log file in the current directory.
"""

import os
import re
import sys
import time
from datetime import datetime
import glob

def find_latest_log():
    """Find the latest log file in the current directory."""
    log_files = glob.glob("*.log")
    if not log_files:
        log_files = glob.glob("../logs/*.log")
    
    if not log_files:
        return None
    
    return max(log_files, key=os.path.getmtime)

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_time(seconds):
    """Format seconds to a readable time format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def create_progress_bar(progress, width=50):
    """Create a textual progress bar."""
    filled_length = int(width * progress)
    bar = '█' * filled_length + '░' * (width - filled_length)
    percentage = progress * 100
    return f"{bar} {percentage:.1f}%"

def parse_log(log_path):
    """Parse training log and extract progress information."""
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Extract relevant information
    data = {
        'start_time': None,
        'latest_time': None,
        'phase': 'unknown',
        'dataset_size': None,
        'samples_processed': 0,
        'total_batches': 0,
        'batch_processed': 0,
        'embedding_complete': False,
        'current_dimension': None,
        'dimensions': [],
        'epochs': 0,
        'current_epoch': 0,
        'train_loss': None,
        'val_loss': None,
        'eta': None,
        'elapsed': None,
        'best_val_loss': float('inf'),
    }
    
    dimension_pattern = re.compile(r'Training distiller for dimensions: (\d+) -> (\d+)')
    epoch_pattern = re.compile(r'Epoch (\d+)/(\d+): train_loss=([\d\.]+), val_loss=([\d\.]+).*time=([\d\.]+)m, ETA=([\d\.]+)m')
    embedding_start_pattern = re.compile(r'Generating embeddings for (\d+) samples \((\d+) batches\)')
    embedding_progress_pattern = re.compile(r'Processing batch (\d+)/(\d+): (\d+) samples \((\d+\.\d+)%\).*ETA: ([\d\.]+)s')
    
    for line in lines:
        # Extract timestamp
        timestamp_match = re.search(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            if data['start_time'] is None:
                data['start_time'] = timestamp
            data['latest_time'] = timestamp
        
        # Check for dataset loading
        if "Loading dataset" in line:
            data['phase'] = 'loading_dataset'
        
        # Check for embedding generation start
        embedding_start_match = embedding_start_pattern.search(line)
        if embedding_start_match:
            data['phase'] = 'embedding_generation'
            data['dataset_size'] = int(embedding_start_match.group(1))
            data['total_batches'] = int(embedding_start_match.group(2))
        
        # Check for embedding generation progress
        embedding_progress_match = embedding_progress_pattern.search(line)
        if embedding_progress_match:
            data['batch_processed'] = int(embedding_progress_match.group(1))
            data['total_batches'] = int(embedding_progress_match.group(2))
            data['samples_processed'] = int(embedding_progress_match.group(3))
            data['eta'] = float(embedding_progress_match.group(5))
        
        # Check for embedding generation completion
        if "Finished generating embeddings" in line:
            data['embedding_complete'] = True
            data['phase'] = 'training_preparation'
        
        # Check for dimension training
        dim_match = dimension_pattern.search(line)
        if dim_match:
            data['phase'] = 'training'
            source_dim = int(dim_match.group(1))
            target_dim = int(dim_match.group(2))
            if [source_dim, target_dim] not in data['dimensions']:
                data['dimensions'].append([source_dim, target_dim])
            data['current_dimension'] = [source_dim, target_dim]
        
        # Check for epoch progress
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            data['current_epoch'] = int(epoch_match.group(1))
            data['epochs'] = int(epoch_match.group(2))
            data['train_loss'] = float(epoch_match.group(3))
            data['val_loss'] = float(epoch_match.group(4))
            data['elapsed'] = float(epoch_match.group(5))
            data['eta'] = float(epoch_match.group(6))
            
            # Track best validation loss
            if data['val_loss'] < data['best_val_loss']:
                data['best_val_loss'] = data['val_loss']
        
        # Check for training completion
        if "Saved model configuration to" in line:
            if data['current_dimension'] == data['dimensions'][-1]:
                data['phase'] = 'complete'
    
    return data

def display_progress(data):
    """Display progress information in a visually appealing way."""
    if not data:
        return
    
    clear_screen()
    print("=" * 80)
    print("DIMENSIONAL CASCADE TRAINING MONITOR")
    print("=" * 80)
    
    # Calculate overall elapsed time
    if data['start_time'] and data['latest_time']:
        elapsed_time = (data['latest_time'] - data['start_time']).total_seconds()
        elapsed_str = format_time(elapsed_time)
    else:
        elapsed_str = "Unknown"
    
    print(f"Elapsed time: {elapsed_str}")
    print(f"Current phase: {data['phase'].replace('_', ' ').title()}")
    print("-" * 80)
    
    # Show phase-specific information
    if data['phase'] == 'embedding_generation':
        if data['total_batches'] > 0:
            progress = data['batch_processed'] / data['total_batches']
            print(f"Embedding Generation Progress:")
            print(create_progress_bar(progress))
            print(f"Batches: {data['batch_processed']}/{data['total_batches']}")
            print(f"Samples: {data['samples_processed']}/{data['dataset_size']}")
            if data['eta']:
                print(f"ETA: {format_time(data['eta'])}")
    
    elif data['phase'] == 'training':
        # Show overall training progress
        total_epochs = len(data['dimensions']) * data['epochs']
        completed_epochs = sum([data['epochs'] for dim in data['dimensions'][:data['dimensions'].index(data['current_dimension'])]])
        completed_epochs += data['current_epoch']
        
        overall_progress = completed_epochs / total_epochs if total_epochs > 0 else 0
        print(f"Overall Training Progress:")
        print(create_progress_bar(overall_progress))
        print(f"Epochs: {completed_epochs}/{total_epochs}")
        
        # Show current dimension progress
        if data['current_dimension'] and data['epochs'] > 0:
            dim_progress = data['current_epoch'] / data['epochs']
            source_dim, target_dim = data['current_dimension']
            print(f"\nCurrent Distillation: {source_dim}D → {target_dim}D")
            print(create_progress_bar(dim_progress))
            print(f"Epoch: {data['current_epoch']}/{data['epochs']}")
            
            if data['train_loss'] is not None and data['val_loss'] is not None:
                print(f"Train Loss: {data['train_loss']:.6f}")
                print(f"Val Loss: {data['val_loss']:.6f}")
                print(f"Best Val Loss: {data['best_val_loss']:.6f}")
            
            if data['eta'] is not None:
                print(f"ETA: {format_time(data['eta'] * 60)}")  # Convert minutes to seconds
        
        # Show dimensions to be trained
        print("\nDimension Plan:")
        for i, (source, target) in enumerate(data['dimensions']):
            if [source, target] == data['current_dimension']:
                status = "🔄 IN PROGRESS"
            elif i < data['dimensions'].index(data['current_dimension']):
                status = "✅ COMPLETE"
            else:
                status = "⏳ PENDING"
            print(f"  {source}D → {target}D: {status}")
    
    elif data['phase'] == 'complete':
        print("Training complete! 🎉")
        print(f"Trained dimensions: {', '.join([f'{s}→{t}' for s, t in data['dimensions']])}")
        print(f"Best validation loss: {data['best_val_loss']:.6f}")
    
    print("\n" + "=" * 80)
    print(f"Last update: {data['latest_time'].strftime('%Y-%m-%d %H:%M:%S') if data['latest_time'] else 'Unknown'}")
    print("=" * 80)

def main():
    """Main function to monitor training logs."""
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = find_latest_log()
        if not log_path:
            print("No log file found. Please specify a log file path.")
            return
    
    print(f"Monitoring log file: {log_path}")
    time.sleep(1)
    
    try:
        while True:
            data = parse_log(log_path)
            display_progress(data)
            
            if data and data['phase'] == 'complete':
                print("Training has completed. Press Ctrl+C to exit.")
            
            time.sleep(5)  # Update every 5 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main() 