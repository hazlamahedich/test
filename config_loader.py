import os
import yaml
from typing import Dict, Any, Optional
import logging

class ConfigurationLoader:
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize configuration loader
        
        :param config_path: Path to the YAML configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        :return: Parsed configuration dictionary
        """
        try:
            # Check if config file exists
            if not os.path.exists(self.config_path):
                self.logger.warning(f"Configuration file not found at {self.config_path}. Using default settings.")
                return {}

            # Load YAML configuration
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            return config or {}
        
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return {}

    def get_global_instructions(self) -> str:
        """
        Retrieve global instructions for content generation
        
        :return: Consolidated global instructions
        """
        global_config = self.config.get('global_instructions', {})
        
        instructions = []
        if global_config.get('tone'):
            instructions.append(f"Maintain a {global_config['tone']} tone.")
        if global_config.get('style'):
            instructions.append(f"Follow a {global_config['style']} writing style.")
        if global_config.get('additional_context'):
            instructions.append(global_config['additional_context'])
        
        return " ".join(instructions) if instructions else ""

    def get_platform_instructions(self, platform: str) -> Dict[str, Any]:
        """
        Retrieve platform-specific instructions
        
        :param platform: Target social media platform
        :return: Dictionary of platform-specific configuration
        """
        platform_config = self.config.get('platform_instructions', {}).get(platform, {})
        
        # Combine global and platform-specific instructions
        global_instructions = self.get_global_instructions()
        platform_instructions = platform_config.get('specific_instructions', '')
        
        full_instructions = f"{global_instructions} {platform_instructions}".strip()
        
        return {
            'instructions': full_instructions,
            'max_length': platform_config.get('max_length'),
            'tone': platform_config.get('tone')
        }

# Create a global instance for easy access
config_loader = ConfigurationLoader()