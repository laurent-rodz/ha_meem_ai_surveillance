import os
import yaml
import re
from dotenv import load_dotenv

# Load .env file at the start
load_dotenv()

def load_config(config_path):
    """
    Load a YAML configuration file and resolve environment variables.
    Supports both ${VAR} placeholders and direct environment overrides for specific keys.
    """
    if not os.path.exists(config_path):
        return {}
        
    with open(config_path, 'r') as f:
        content = f.read()
        
        # 1. Resolve ${VAR} placeholders if they exist
        pattern = re.compile(r'\$\{(\w+)(?::([^}]*))?\}')
        def replace_match(match):
            var_name = match.group(1)
            default_value = match.group(2)
            return os.getenv(var_name, default_value if default_value is not None else match.group(0))
        
        resolved_content = pattern.sub(replace_match, content)
        config = yaml.safe_load(resolved_content) or {}

        # 2. Automatic Overrides for critical keys (even without placeholders)
        # This allows the YAML to stay hardcoded while .env takes precedence
        _apply_env_overrides(config)
        
        return config

def _apply_env_overrides(config):
    """Recursively check for environment variable overrides."""
    # Mapping of specific critical ENV vars to their YAML paths
    overrides = {
        "CAMERA_01_URL": ("cameras", 0, "url"),
        "CAMERA_02_URL": ("cameras", 1, "url"),
    }

    for env_var, path in overrides.items():
        value = os.getenv(env_var)
        if value:
            # Traverse the config dictionary to apply the override
            current = config
            for i, step in enumerate(path):
                if i == len(path) - 1:
                    if isinstance(current, (dict, list)) and step < len(current) if isinstance(current, list) else step in current:
                        current[step] = value
                else:
                    try:
                        current = current[step]
                    except (KeyError, IndexError, TypeError):
                        break

def load_merged_configs(paths):
    """Load and merge multiple configuration files."""
    merged_config = {}
    for path in paths:
        config = load_config(path)
        if config:
            for key, value in config.items():
                if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
    return merged_config
