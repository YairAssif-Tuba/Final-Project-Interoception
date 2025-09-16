"""Utility functions for model management.

This module provides essential utilities for managing neural network models:
1. Loading and saving hyperparameters
2. Creating directories for model output
3. Managing training logs

ENHANCED: Support for piezo-enhanced models
BACKWARD COMPATIBLE: All original functionality preserved

These functions support the interval timing tasks (interval_production and interval_comparison)
by providing a consistent interface for model configuration and result storage.
"""

import os
import errno
import json
import numpy as np

import default


def validate_piezo_hp(hp):
    """Validate piezo-specific hyperparameters."""
    if not hp.get('use_piezo', False):
        return  # No validation needed if piezo is disabled

    # SIMPLIFIED: Only check for simplified piezo parameters
    required_keys = [
        'heartbeat_slice_size',
        'piezo_connection_fraction'
        # Removed num_harmonics and signal_feature_count
    ]

    for key in required_keys:
        if key not in hp:
            raise ValueError(f"Missing required piezo parameter: {key}")

    # Validate ranges
    if not (0.0 < hp['piezo_connection_fraction'] <= 1.0):
        raise ValueError(f"piezo_connection_fraction must be in (0, 1], got {hp['piezo_connection_fraction']}")

    if hp['heartbeat_slice_size'] < 1:
        raise ValueError(f"heartbeat_slice_size must be >= 1, got {hp['heartbeat_slice_size']}")

    print(f"✅ Simplified piezo hyperparameters validated")

def load_hp(model_dir):
    """Load hyperparameters from a model directory.
    
    Args:
        model_dir: Path to the model directory containing hp.json
        
    Returns:
        hp: Dictionary of hyperparameters
        
    Note:
        Sets a new random seed after loading, as loading is typically done
        for analysis rather than continued training.
    """
    fname = os.path.join(model_dir, 'hp.json')

    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            hp = json.load(f)
    else:
        print(f"Warning: Hyperparameter file not found at {fname}")
        hp = default.get_default_hp()

    # Use a different seed after loading since loading is typically for analysis
    hp['seed'] = np.random.randint(0, 1000000)
    hp['rng'] = np.random.RandomState(hp['seed'])
    
    # Validate piezo parameters if enabled
    try:
        validate_piezo_hp(hp)
    except ValueError as e:
        print(f"⚠️ Piezo validation warning: {e}")
        print("   Model may not function correctly with piezo interface")
    
    return hp


def save_hp(hp, model_dir):
    """Save hyperparameters to a model directory.
    
    Args:
        hp: Dictionary of hyperparameters
        model_dir: Path to the model directory where hp.json will be saved
    """
    hp_copy = hp.copy()
    hp_copy.pop('rng', None)  # rng cannot be serialized
    
    # Validate before saving if piezo is enabled
    if hp_copy.get('use_piezo', False):
        try:
            validate_piezo_hp(hp_copy)
        except ValueError as e:
            print(f"⚠️ Warning: Saving potentially invalid piezo hyperparameters: {e}")
    
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f, indent=2)


def mkdir_p(path):
    """Create directory if it doesn't exist (portable mkdir -p).
    
    Args:
        path: Directory path to create
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_log(log, model_dir=None, log_name='log.json'):
    """Save training log to a file.
    
    Args:
        log: Dictionary containing log information
        model_dir: Directory to save log (if None, uses log['model_dir'])
        log_name: Name of the log file (default: 'log.json')
    """
    if model_dir is None:
        model_dir = log.get('model_dir', '.')
        
    fname = os.path.join(model_dir, log_name)
    
    # Create a copy to avoid modifying original
    log_copy = dict(log)
    
    # Add timestamp
    import time
    log_copy['save_timestamp'] = time.time()
    
    with open(fname, 'w') as f:
        json.dump(log_copy, f, indent=2)


def load_log(model_dir, log_name='log.json'):
    """Load training log from a file.
    
    Args:
        model_dir: Path to the model directory containing the log file
        log_name: Name of the log file (default: 'log.json')
        
    Returns:
        log: Dictionary containing log information, or None if file doesn't exist
    """
    fname = os.path.join(model_dir, log_name)
    if not os.path.isfile(fname):
        return None

    with open(fname, 'r') as f:
        log = json.load(f)
    return log


def save_pretraining_checkpoint(model, model_dir, phase='pretraining', 
                               additional_info=None):
    """Save model checkpoint during pretraining phase.
    
    Args:
        model: The neural network model
        model_dir: Base model directory
        phase: Phase name (default: 'pretraining')
        additional_info: Additional information to save with checkpoint
    """
    checkpoint_dir = os.path.join(model_dir, f'{phase}_checkpoint')
    mkdir_p(checkpoint_dir)
    
    # Save model
    model.save(checkpoint_dir)
    
    # Save additional info if provided
    if additional_info is not None:
        info_path = os.path.join(checkpoint_dir, f'{phase}_info.json')
        with open(info_path, 'w') as f:
            json.dump(additional_info, f, indent=2)
    
    print(f"💾 Saved {phase} checkpoint to {checkpoint_dir}")


def load_model_with_fallback(model_dir, hp, is_cuda=False, **kwargs):
    """Load model with fallback to different checkpoint directories.
    
    Args:
        model_dir: Primary model directory
        hp: Hyperparameters
        is_cuda: Whether to use CUDA
        **kwargs: Additional arguments for model creation
        
    Returns:
        model: Loaded model
        checkpoint_info: Information about which checkpoint was loaded
    """
    import network
    
    # Try different checkpoint directories in order of preference
    checkpoint_dirs = [
        model_dir,  # Main directory
        os.path.join(model_dir, 'finalResult'),  # Final result
        os.path.join(model_dir, 'pretraining_checkpoint'),  # Pretraining
    ]
    
    # Add numbered checkpoints
    for i in range(10):
        checkpoint_dirs.append(os.path.join(model_dir, str(i)))
    
    model = None
    loaded_from = None
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            try:
                temp_model = network.RNN(hp, is_cuda, **kwargs)
                if temp_model.load(checkpoint_dir):
                    model = temp_model
                    loaded_from = checkpoint_dir
                    break
            except Exception as e:
                print(f"⚠️ Failed to load from {checkpoint_dir}: {e}")
                continue
    
    if model is None:
        print(f"🔧 Creating new model (no valid checkpoints found)")
        model = network.RNN(hp, is_cuda, **kwargs)
        loaded_from = "new_model"
    else:
        print(f"✅ Loaded model from {loaded_from}")
    
    checkpoint_info = {
        'loaded_from': loaded_from,
        'available_checkpoints': [d for d in checkpoint_dirs if os.path.exists(d)]
    }
    
    return model, checkpoint_info


def analyze_model_directory(model_dir):
    """Analyze a model directory and return information about available files.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        dict: Analysis of model directory contents
    """
    analysis = {
        'model_dir': model_dir,
        'exists': os.path.exists(model_dir),
        'has_hp': False,
        'has_model': False,
        'has_log': False,
        'checkpoints': [],
        'use_piezo': False,
        'pretraining_checkpoint': False
    }
    
    if not analysis['exists']:
        return analysis
    
    # Check for main files
    analysis['has_hp'] = os.path.exists(os.path.join(model_dir, 'hp.json'))
    analysis['has_model'] = os.path.exists(os.path.join(model_dir, 'model.pth'))
    analysis['has_log'] = os.path.exists(os.path.join(model_dir, 'log.json'))
    
    # Check for piezo configuration
    if analysis['has_hp']:
        try:
            hp = load_hp(model_dir)
            analysis['use_piezo'] = hp.get('use_piezo', False)
        except:
            pass
    
    # Find checkpoint directories
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path):
            if item == 'pretraining_checkpoint':
                analysis['pretraining_checkpoint'] = True
            elif item == 'finalResult':
                analysis['checkpoints'].append({'name': item, 'type': 'final'})
            elif item.isdigit():
                analysis['checkpoints'].append({'name': item, 'type': 'numbered', 'number': int(item)})
            else:
                analysis['checkpoints'].append({'name': item, 'type': 'other'})
    
    # Sort numbered checkpoints
    analysis['checkpoints'].sort(key=lambda x: (x['type'], x.get('number', 0)))
    
    return analysis


def print_model_summary(model_dir):
    """Print a summary of a model directory.
    
    Args:
        model_dir: Path to model directory
    """
    analysis = analyze_model_directory(model_dir)
    
    print(f"\n📂 Model Directory: {model_dir}")
    print("=" * 50)
    
    if not analysis['exists']:
        print("❌ Directory does not exist")
        return
    
    # Basic files
    print(f"📄 Hyperparameters: {'✅' if analysis['has_hp'] else '❌'}")
    print(f"🤖 Model weights: {'✅' if analysis['has_model'] else '❌'}")
    print(f"📊 Training log: {'✅' if analysis['has_log'] else '❌'}")
    
    # Piezo status
    if analysis['use_piezo']:
        print(f"🫀 Piezo interface: ✅ ENABLED")
        if analysis['pretraining_checkpoint']:
            print(f"   Pretraining checkpoint: ✅")
        else:
            print(f"   Pretraining checkpoint: ❌")
    else:
        print(f"🫀 Piezo interface: ❌ DISABLED")
    
    # Checkpoints
    if analysis['checkpoints']:
        print(f"💾 Checkpoints ({len(analysis['checkpoints'])}):")
        for checkpoint in analysis['checkpoints'][:5]:  # Show first 5
            print(f"   - {checkpoint['name']} ({checkpoint['type']})")
        if len(analysis['checkpoints']) > 5:
            print(f"   ... and {len(analysis['checkpoints']) - 5} more")
    else:
        print(f"💾 Checkpoints: None")


def migrate_legacy_model(model_dir, backup=True):
    """Migrate a legacy (non-piezo) model directory to support piezo.
    
    Args:
        model_dir: Path to legacy model directory
        backup: Whether to create backup of original files
        
    Returns:
        bool: True if migration was successful
    """
    analysis = analyze_model_directory(model_dir)
    
    if not analysis['exists'] or not analysis['has_hp']:
        print(f"❌ Cannot migrate: Invalid model directory")
        return False
        
    if analysis['use_piezo']:
        print(f"ℹ️ Model already supports piezo interface")
        return True
    
    print(f"🔄 Migrating legacy model to support piezo interface...")
    
    try:
        # Load existing hyperparameters
        hp = load_hp(model_dir)
        
        # Create backup if requested
        if backup:
            backup_dir = model_dir + '_backup'
            if not os.path.exists(backup_dir):
                import shutil
                shutil.copytree(model_dir, backup_dir)
                print(f"💾 Backup created: {backup_dir}")
        
        # Add piezo parameters (disabled by default)
        hp['use_piezo'] = False
        hp['heartbeat_slice_size'] = 20
        hp['num_harmonics'] = 5
        hp['piezo_connection_fraction'] = 0.15
        hp['use_temporal_delay'] = False
        hp['signal_feature_count'] = 15
        hp['pretraining_epochs'] = 100
        hp['pretraining_lr'] = 0.001
        hp['pretraining_num_samples'] = 100
        hp['pretraining_T'] = 50
        
        # Save updated hyperparameters
        save_hp(hp, model_dir)
        
        print(f"✅ Migration complete - piezo support added (disabled by default)")
        print(f"   To enable piezo: set 'use_piezo': true in hp.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False


def cleanup_model_directory(model_dir, keep_best=3, keep_final=True):
    """Clean up model directory by removing old checkpoints.
    
    Args:
        model_dir: Path to model directory
        keep_best: Number of best checkpoints to keep
        keep_final: Whether to keep finalResult directory
        
    Returns:
        int: Number of checkpoints removed
    """
    analysis = analyze_model_directory(model_dir)
    
    if not analysis['exists']:
        print(f"❌ Directory does not exist: {model_dir}")
        return 0
    
    # Find numbered checkpoints
    numbered_checkpoints = [c for c in analysis['checkpoints'] if c['type'] == 'numbered']
    
    if len(numbered_checkpoints) <= keep_best:
        print(f"ℹ️ No cleanup needed ({len(numbered_checkpoints)} <= {keep_best} checkpoints)")
        return 0
    
    # Sort by number (newest first) and keep only the best
    numbered_checkpoints.sort(key=lambda x: x['number'], reverse=True)
    to_remove = numbered_checkpoints[keep_best:]
    
    removed_count = 0
    for checkpoint in to_remove:
        checkpoint_path = os.path.join(model_dir, checkpoint['name'])
        try:
            import shutil
            shutil.rmtree(checkpoint_path)
            removed_count += 1
            print(f"🗑️ Removed checkpoint: {checkpoint['name']}")
        except Exception as e:
            print(f"⚠️ Failed to remove {checkpoint['name']}: {e}")
    
    print(f"✅ Cleanup complete: removed {removed_count} old checkpoints")
    return removed_count