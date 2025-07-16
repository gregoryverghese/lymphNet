"""
Configuration management utilities for LymphNet
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages configuration and paths for LymphNet"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file"""
        save_path = path or self.config_path
        if save_path:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)


class PathManager:
    """Manages file and directory paths for LymphNet"""
    
    def __init__(self, base_path: str = "."):
        """
        Initialize path manager
        
        Args:
            base_path: Base directory for the project
        """
        self.base_path = Path(base_path).resolve()
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup common project paths"""
        self.paths = {
            'base': self.base_path,
            'src': self.base_path / 'src',
            'config': self.base_path / 'config',
            'data': self.base_path / 'data',
            'output': self.base_path / 'output',
            'models': self.base_path / 'models',
            'logs': self.base_path / 'logs',
            'tests': self.base_path / 'tests',
        }
    
    def get_path(self, name: str) -> Path:
        """Get path by name"""
        return self.paths.get(name, self.base_path / name)
    
    def create_dirs(self, *dir_names: str):
        """Create directories if they don't exist"""
        for name in dir_names:
            path = self.get_path(name)
            path.mkdir(parents=True, exist_ok=True)
    
    def resolve_path(self, path: str) -> Path:
        """Resolve relative path to absolute"""
        if os.path.isabs(path):
            return Path(path)
        return self.base_path / path


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_file: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_file: Path to save configuration
    """
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent.parent


def setup_experiment_paths(experiment_name: str, base_output_dir: str = "output") -> Dict[str, Path]:
    """
    Setup paths for an experiment
    
    Args:
        experiment_name: Name of the experiment
        base_output_dir: Base output directory
        
    Returns:
        Dictionary of experiment paths
    """
    project_root = get_project_root()
    experiment_base = project_root / base_output_dir / experiment_name
    
    paths = {
        'base': experiment_base,
        'models': experiment_base / 'models',
        'curves': experiment_base / 'curves',
        'predictions': experiment_base / 'predictions',
        'logs': experiment_base / 'logs',
        'tensorboard': experiment_base / 'tensorboard_logs',
    }
    
    # Create directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths 