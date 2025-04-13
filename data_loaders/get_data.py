"""
Data loading utilities for motion datasets.

This module provides a unified interface for loading different types of motion datasets:
- HumanML3D: Basic human motion dataset
- HumanML3D+: Extended dataset with BABEL annotations
"""

from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate, t2m_collate_w_name
from data_loaders.tensors import t2m_collate, t2m_collate_mix, t2m_collate_hml
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatasetConfig:
    """Configuration for dataset loading.
    
    This class centralizes all parameters needed for dataset and dataloader creation.
    It helps maintain consistency across different dataset variants and makes it
    easier to add new configuration options.
    
    Attributes:
        name: Dataset name ('humanml' or 'humanml+')
        num_frames: Number of frames to load per sequence
        split: Dataset split ('train', 'val', 'test', etc.)
        hml_mode: Loading mode ('train', 'eval', 'gt')
        pca: PCA components to use (0 for no PCA)
        size: Optional size limit for dataset
        batch_size: Batch size for DataLoader
        args: Additional arguments passed to dataset
    """
    name: str
    num_frames: int
    split: str = 'train'
    hml_mode: str = 'train'
    pca: int = 0
    size: Optional[int] = None
    batch_size: Optional[int] = None
    args: Optional[object] = None

def get_dataset_class(config: DatasetConfig):
    """Get the appropriate dataset class based on configuration.
    
    This function handles the complexity of selecting the correct dataset class
    based on the configuration. For HumanML3D+, it considers:
    - Whether we're using BABEL annotations
    - Whether we're in training mode
    - Which mean/std normalization to use
    
    Args:
        config: DatasetConfig object containing dataset parameters
        
    Returns:
        Dataset class to instantiate
        
    Raises:
        ValueError: If dataset name is not supported
    """
    if config.name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
        
    if config.name == "humanml+":
        from data_loaders.humanml.data.dataset import (
            HumanML3D_Babel_union,  # Uses HML+BABEL mean/std
            Babel_full,      # Uses BABEL mean/std
            HumanML3D_Full,        # Uses full HML mean/std
            HumanML3D_BABEL_overlap            # Uses HumanML^Babel overlap mean/std
        )
        
        # Training with BABEL dataset with the sequences only have local label annatations - uses different normalizations for train/test
        if hasattr(config.args, 'train_babel') and config.args.train_babel:
            return HumanML3D_Babel_union if config.split == 'train' else Babel_full
            
        # Training with full HumanML3D also includes without local label sequences
        if hasattr(config.args, 'train_split') and config.split == 'train':
            return HumanML3D_Full
            
        return HumanML3D_BABEL_overlap
        
    raise ValueError(f'Unsupported dataset name [{config.name}]')

def determine_collate_settings(config: DatasetConfig):
    """Determine collate function settings based on configuration.
    
    This function handles the logic for selecting the appropriate collate function
    and its settings. It considers:
    - Whether we need motion names (IDs)
    - Whether text labels are mix global/local labels
    - Whether is the full HumanML3D dataset
    
    Args:
        config: DatasetConfig object
        
    Returns:
        tuple: (collate_function, settings_dict)
    """
    settings = {
        'need_name': False,  # Whether to include motion IDs
        'mix': False,        # Whether text labels are mix global/local labels
        'hml': False         # Whether is the full HumanML3D dataset
    }
    
    # Ground truth mode uses a special collate function
    if config.hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate, settings
    
    # Non-standard datasets use the basic collate
    if config.name not in ["humanml", "humanml+"]:
        return all_collate, settings
        
    # Check if we need motion IDs based on split type
    no_overlap_splits = ['test_ft_no_overlap', 'test_val_no_overlap', 'val_no_overlap']
    settings['need_name'] = any(split in config.split for split in no_overlap_splits)
    
    # Configure BABEL and HML settings
    if hasattr(config.args, 'train_babel') and config.args.train_babel:
        settings.update({'need_name': True, 'mix': True})  # BABEL needs both name and mix
    elif config.split == 'train':
        settings.update({'need_name': True, 'hml': True})  # Training needs name and HML
        
    # Select appropriate collate function based on settings
    if settings['need_name']:
        if settings['mix']:
            return t2m_collate_mix, settings      # For mixed global/local labels
        if settings['hml']:
            return t2m_collate_hml, settings      # For HML features
        return t2m_collate_w_name, settings       # For basic named collation
        
    return t2m_collate, settings                  # Basic collation without names

def get_dataset(config: DatasetConfig):
    """Get dataset instance based on configuration.
    
    Creates and configures a dataset instance with the appropriate parameters.
    Handles special cases for certain dataset types that need extra arguments.
    
    Args:
        config: DatasetConfig object
        
    Returns:
        Dataset instance
    """
    DatasetClass = get_dataset_class(config)
    
    # Common arguments for all dataset types
    common_args = {
        'split': config.split,
        'num_frames': config.num_frames,
        'mode': config.hml_mode,
        'pca': config.pca,
        'size': config.size,
    }
    
    # Add extra args for specific datasets
    if config.name in ["humanml", "humanml+"] and (
        (hasattr(config.args, 'train_babel') and config.args.train_babel) or
        (hasattr(config.args, 'train_split') and config.split == 'train')
    ):
        common_args['args'] = config.args
        
    return DatasetClass(**common_args)

def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', pca=0, size=None, args=None):
    """Get DataLoader instance based on parameters.
    
    Creates a DataLoader with appropriate settings for training or evaluation.
    Handles shuffle and drop_last settings based on the mode.
    
    Args:
        name: Dataset name ('humanml' or 'humanml+')
        batch_size: Batch size for DataLoader
        num_frames: Number of frames to load per sequence
        split: Dataset split ('train', 'val', 'test', etc.)
        hml_mode: Loading mode ('train', 'eval', 'gt')
        pca: PCA components to use (0 for no PCA)
        size: Optional size limit for dataset
        args: Additional arguments passed to dataset
        
    Returns:
        DataLoader instance
    """
    config = DatasetConfig(
        name=name,
        batch_size=batch_size,
        num_frames=num_frames,
        split=split,
        hml_mode=hml_mode,
        pca=pca,
        size=size,
        args=args
    )
    
    dataset = get_dataset(config)
    collate_fn, _ = determine_collate_settings(config)
    
    # Don't shuffle or drop last batch in eval/gt modes
    shuffle = hml_mode not in ['eval', 'gt']
    drop_last = hml_mode not in ['eval', 'gt']
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        drop_last=drop_last,
        collate_fn=collate_fn
    )

def get_dataset_loader_eval(name, batch_size, num_frames, split='train', hml_mode='train', pca=0, size=None, args=None):
    """Legacy support function for evaluation data loading.
    
    This function maintains backward compatibility with older code that uses
    individual parameters instead of the new config-based approach.
    """
    config = DatasetConfig(
        name=name,
        batch_size=batch_size,
        num_frames=num_frames,
        split=split,
        hml_mode=hml_mode,
        pca=pca,
        size=size,
        args=args
    )
    return get_dataset_loader(name, batch_size, num_frames, split, hml_mode, pca, size, args)