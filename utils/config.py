import yaml
import argparse
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='USSL Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device(s) to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()


def setup_experiment(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Setup experiment configuration."""
    # Update config with command line arguments
    config['experiment'] = {
        'seed': args.seed,
        'gpu': args.gpu,
        'debug': args.debug,
        'resume': args.resume
    }
    
    # Create directories
    Path(config['logging']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['checkpoint']['save_dir']).mkdir(parents=True, exist_ok=True)
    
    return config 