"""
Domain-Specific Configurations for SeededSphere
Provides optimized presets for different types of content.
"""

import json
import os
from typing import Dict, Optional, List

class DomainConfigurator:
    """Manages domain-specific configurations for SeededSphere search"""
    
    # Default configuration presets for different domains
    DEFAULT_PRESETS = {
        "academic": {
            "min_echo_layers": 2,
            "max_echo_layers": 6,
            "delta": 0.15,
            "alpha": 0.6,
            "importance_words": [
                "cited", "paper", "research", "findings", "study",
                "methodology", "analysis", "results", "conclusion"
            ],
            "preprocessing": {
                "remove_citations": True,
                "normalize_equations": True,
                "preserve_sections": True
            }
        },
        "code": {
            "min_echo_layers": 1,
            "max_echo_layers": 4,
            "delta": 0.2,
            "alpha": 0.4,
            "importance_words": [
                "function", "class", "method", "return", "import",
                "variable", "parameter", "interface", "implementation"
            ],
            "preprocessing": {
                "remove_comments": False,
                "normalize_whitespace": True,
                "preserve_indentation": True
            }
        },
        "general": {
            "min_echo_layers": 2,
            "max_echo_layers": 5,
            "delta": 0.1625,
            "alpha": 0.5,
            "importance_words": [],
            "preprocessing": {
                "remove_citations": False,
                "normalize_whitespace": True,
                "preserve_sections": False
            }
        }
    }
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize domain configurator.
        
        Args:
            config_dir: Optional directory for custom configuration files
        """
        self.config_dir = config_dir or "configs"
        self.custom_presets = {}
        
        # Load any custom configurations
        self._load_custom_configs()
    
    def _load_custom_configs(self):
        """Load custom configuration files from config directory"""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            return
            
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                domain = filename[:-5]  # Remove .json extension
                filepath = os.path.join(self.config_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        self.custom_presets[domain] = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON in {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def get_configuration(self, domain: str = "general", sample_text: Optional[str] = None) -> Dict:
        """
        Get configuration for specified domain.
        
        Args:
            domain: Domain name ("academic", "code", "general", or custom)
            sample_text: Optional text sample for domain detection
            
        Returns:
            Configuration dictionary
        """
        if domain == "auto" and sample_text:
            domain = self.detect_domain(sample_text)
        
        # Check custom presets first
        if domain in self.custom_presets:
            config = self.custom_presets[domain].copy()
        # Then check default presets
        elif domain in self.DEFAULT_PRESETS:
            config = self.DEFAULT_PRESETS[domain].copy()
        else:
            print(f"Warning: Unknown domain '{domain}', using general configuration")
            config = self.DEFAULT_PRESETS["general"].copy()
        
        return config
    
    def detect_domain(self, text: str) -> str:
        """
        Detect domain from text content.
        
        Args:
            text: Sample text content
            
        Returns:
            Detected domain name
        """
        # Simple keyword-based detection
        text = text.lower()
        
        # Count domain-specific keywords
        scores = {}
        
        for domain, preset in {**self.DEFAULT_PRESETS, **self.custom_presets}.items():
            importance_words = preset.get("importance_words", [])
            score = sum(1 for word in importance_words if word in text)
            scores[domain] = score
        
        # Return domain with highest score, defaulting to "general"
        if not scores:
            return "general"
            
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def save_custom_config(self, domain: str, config: Dict):
        """
        Save custom configuration for a domain.
        
        Args:
            domain: Domain name
            config: Configuration dictionary
        """
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            
        filepath = os.path.join(self.config_dir, f"{domain}.json")
        
        # Validate required fields
        required_fields = ["min_echo_layers", "max_echo_layers", "delta", "alpha"]
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Save configuration
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
            
        # Update custom presets
        self.custom_presets[domain] = config
        
        print(f"Custom configuration for domain '{domain}' saved to {filepath}")
    
    def list_domains(self) -> List[str]:
        """
        List all available domain configurations.
        
        Returns:
            List of domain names
        """
        domains = list(self.DEFAULT_PRESETS.keys()) + \
                 [d for d in self.custom_presets.keys() if d not in self.DEFAULT_PRESETS]
        return sorted(domains)
    
    def get_domain_info(self, domain: str) -> Dict:
        """
        Get detailed information about a domain configuration.
        
        Args:
            domain: Domain name
            
        Returns:
            Dictionary with domain configuration details
        """
        config = self.get_configuration(domain)
        
        return {
            "name": domain,
            "source": "custom" if domain in self.custom_presets else "default",
            "config": config,
            "description": self._get_domain_description(domain)
        }
    
    def _get_domain_description(self, domain: str) -> str:
        """Get description for a domain"""
        descriptions = {
            "academic": "Optimized for academic papers and research documents. "
                       "Preserves citation context and section structure.",
            "code": "Tailored for source code and technical documentation. "
                   "Maintains code structure and identifier relationships.",
            "general": "Balanced configuration for general web content and documents. "
                      "Provides good performance across various content types."
        }
        return descriptions.get(domain, "Custom domain configuration")
