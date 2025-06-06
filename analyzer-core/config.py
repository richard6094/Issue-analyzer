"""
Configuration Management for Intelligent Function Dispatcher

This module handles configuration loading, validation, and management
for the GitHub Issue Agent intelligent dispatcher system.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: str = "azure_openai"  # or "openai"
    model: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 30
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None


@dataclass
class RAGConfig:
    """Configuration for RAG (Retrieval Augmented Generation)"""
    enabled: bool = True
    vector_db_path: str = "data/vector_db"
    similarity_threshold: float = 0.7
    max_results: int = 5
    embedding_model: str = "text-embedding-ada-002"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class ImageAnalysisConfig:
    """Configuration for image analysis"""
    enabled: bool = True
    provider: str = "azure_computer_vision"
    max_image_size_mb: int = 10
    supported_formats: List[str] = field(default_factory=lambda: ["png", "jpg", "jpeg", "gif", "svg"])
    timeout: int = 30
    api_key: Optional[str] = None
    api_base: Optional[str] = None


@dataclass
class GitHubConfig:
    """Configuration for GitHub integration"""
    token: Optional[str] = None
    webhook_secret: Optional[str] = None
    api_base_url: str = "https://api.github.com"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_buffer: int = 100  # Reserve buffer for rate limits


@dataclass
class WebhookConfig:
    """Configuration for webhook handling"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    webhook_path: str = "/webhook/github"
    health_check_path: str = "/health"
    test_path: str = "/webhook/test"


@dataclass
class ProcessingConfig:
    """Configuration for processing behavior"""
    max_concurrent_requests: int = 10
    request_timeout: int = 300  # 5 minutes
    enable_async_processing: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    retry_failed_operations: bool = True
    max_operation_retries: int = 3


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_console: bool = True
    enable_structured_logging: bool = False


@dataclass
class IntelligentDispatcherConfig:
    """Main configuration class for the intelligent dispatcher"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    image_analysis: ImageAnalysisConfig = field(default_factory=ImageAnalysisConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Feature flags
    enable_intent_analysis: bool = True
    enable_rag_analysis: bool = True
    enable_image_analysis: bool = True
    enable_regression_analysis: bool = True
    enable_conversation_analysis: bool = True
    enable_template_generation: bool = True
    
    # Trigger conditions
    min_issue_title_length: int = 5
    min_issue_body_length: int = 20
    excluded_bot_users: List[str] = field(default_factory=lambda: [
        "github-actions", "dependabot", "codecov", "bot"
    ])
    
    # Analysis thresholds
    high_urgency_threshold: int = 4
    high_complexity_threshold: int = 4
    confidence_threshold: float = 0.7


class ConfigManager:
    """
    Configuration manager for loading and validating configuration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file (optional)
        """        self.config_path = config_path
        self.config = IntelligentDispatcherConfig()
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from multiple sources"""
        try:
            # 1. Load from file if provided
            if self.config_path and os.path.exists(self.config_path):
                self._load_from_file(self.config_path)
            
            # 2. Load from environment variables
            self._load_from_environment()
            
            # 3. Validate configuration
            self._validate_configuration()
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _load_from_file(self, config_path: str) -> None:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update configuration with file data
            self._update_config_from_dict(config_data)
            logger.info(f"Loaded configuration from file: {config_path}")
            
        except Exception as e:
            logger.warning(f"Could not load config file {config_path}: {str(e)}")
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables"""
        # LLM Configuration
        if os.environ.get('LLM_PROVIDER'):
            self.config.llm.provider = os.environ['LLM_PROVIDER']
        if os.environ.get('LLM_MODEL'):
            self.config.llm.model = os.environ['LLM_MODEL']
        if os.environ.get('LLM_TEMPERATURE'):
            self.config.llm.temperature = float(os.environ['LLM_TEMPERATURE'])
        if os.environ.get('OPENAI_API_KEY'):
            self.config.llm.api_key = os.environ['OPENAI_API_KEY']
        if os.environ.get('AZURE_OPENAI_ENDPOINT'):
            self.config.llm.api_base = os.environ['AZURE_OPENAI_ENDPOINT']
        if os.environ.get('AZURE_OPENAI_API_VERSION'):
            self.config.llm.api_version = os.environ['AZURE_OPENAI_API_VERSION']
        if os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME'):
            self.config.llm.deployment_name = os.environ['AZURE_OPENAI_DEPLOYMENT_NAME']
        
        # GitHub Configuration
        if os.environ.get('GITHUB_TOKEN'):
            self.config.github.token = os.environ['GITHUB_TOKEN']
        if os.environ.get('GITHUB_WEBHOOK_SECRET'):
            self.config.github.webhook_secret = os.environ['GITHUB_WEBHOOK_SECRET']
        
        # Image Analysis Configuration
        if os.environ.get('AZURE_COMPUTER_VISION_KEY'):
            self.config.image_analysis.api_key = os.environ['AZURE_COMPUTER_VISION_KEY']
        if os.environ.get('AZURE_COMPUTER_VISION_ENDPOINT'):
            self.config.image_analysis.api_base = os.environ['AZURE_COMPUTER_VISION_ENDPOINT']
        
        # Webhook Configuration
        if os.environ.get('WEBHOOK_HOST'):
            self.config.webhook.host = os.environ['WEBHOOK_HOST']
        if os.environ.get('WEBHOOK_PORT'):
            self.config.webhook.port = int(os.environ['WEBHOOK_PORT'])
        if os.environ.get('DEBUG'):
            self.config.webhook.debug = os.environ['DEBUG'].lower() == 'true'
        
        # Feature Flags
        if os.environ.get('ENABLE_RAG_ANALYSIS'):
            self.config.enable_rag_analysis = os.environ['ENABLE_RAG_ANALYSIS'].lower() == 'true'
        if os.environ.get('ENABLE_IMAGE_ANALYSIS'):
            self.config.enable_image_analysis = os.environ['ENABLE_IMAGE_ANALYSIS'].lower() == 'true'
        
        # Logging Configuration
        if os.environ.get('LOG_LEVEL'):
            self.config.logging.level = os.environ['LOG_LEVEL']
        if os.environ.get('LOG_FILE_PATH'):
            self.config.logging.file_path = os.environ['LOG_FILE_PATH']
        
        logger.info("Configuration updated from environment variables")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for section_name, section_data in config_data.items():
            if hasattr(self.config, section_name) and isinstance(section_data, dict):
                section_config = getattr(self.config, section_name)
                for key, value in section_data.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
            elif hasattr(self.config, section_name):
                setattr(self.config, section_name, section_data)
    
    def _validate_configuration(self) -> None:
        """Validate configuration settings"""
        errors = []
        
        # Validate required GitHub token
        if not self.config.github.token:
            errors.append("GitHub token is required (set GITHUB_TOKEN environment variable)")
        
        # Validate LLM configuration
        if self.config.llm.provider == "azure_openai":
            if not self.config.llm.api_base:
                errors.append("Azure OpenAI endpoint is required for azure_openai provider")
            if not self.config.llm.deployment_name:
                errors.append("Azure OpenAI deployment name is required for azure_openai provider")
        elif self.config.llm.provider == "openai":
            if not self.config.llm.api_key:
                errors.append("OpenAI API key is required for openai provider")
        
        # Validate numeric ranges
        if not 0 <= self.config.llm.temperature <= 2:
            errors.append("LLM temperature must be between 0 and 2")
        
        if not 0 <= self.config.rag.similarity_threshold <= 1:
            errors.append("RAG similarity threshold must be between 0 and 1")
        
        if self.config.webhook.port < 1 or self.config.webhook.port > 65535:
            errors.append("Webhook port must be between 1 and 65535")
        
        # Validate paths
        if self.config.rag.vector_db_path:
            vector_db_dir = Path(self.config.rag.vector_db_path).parent
            if not vector_db_dir.exists():
                try:
                    vector_db_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create vector DB directory: {str(e)}")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    def get_config(self) -> IntelligentDispatcherConfig:
        """Get the current configuration"""
        return self.config
    
    def save_config(self, output_path: str) -> None:
        """Save current configuration to file"""
        try:
            config_dict = self._config_to_dict()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {}
        
        for field_name in self.config.__dataclass_fields__:
            field_value = getattr(self.config, field_name)
            
            if hasattr(field_value, '__dataclass_fields__'):
                # Nested dataclass
                config_dict[field_name] = {
                    nested_field: getattr(field_value, nested_field)
                    for nested_field in field_value.__dataclass_fields__
                }
            else:
                config_dict[field_name] = field_value
        
        return config_dict
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        try:
            self._update_config_from_dict(updates)
            self._validate_configuration()
            logger.info("Configuration updated successfully")
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            raise
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration for setup"""
        return {
            'level': getattr(logging, self.config.logging.level.upper()),
            'format': self.config.logging.format,
            'filename': self.config.logging.file_path,
            'maxBytes': self.config.logging.max_file_size_mb * 1024 * 1024,
            'backupCount': self.config.logging.backup_count
        }


def setup_logging(config: LoggingConfig) -> None:
    """Setup logging based on configuration"""
    import logging.handlers
    
    # Clear existing handlers
    logging.getLogger().handlers.clear()
    
    # Set logging level
    log_level = getattr(logging, config.level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(config.format)
    
    # Console handler
    if config.enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    
    # File handler
    if config.file_path:
        try:
            # Create directory if it doesn't exist
            log_dir = Path(config.file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                config.file_path,
                maxBytes=config.max_file_size_mb * 1024 * 1024,
                backupCount=config.backup_count
            )
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not setup file logging: {str(e)}")


# Default configuration instance
_default_config_manager = None


def get_config() -> SmartDispatcherConfig:
    """Get the default configuration instance"""
    global _default_config_manager
    
    if _default_config_manager is None:
        # Look for config file in common locations
        config_paths = [
            "config.json",
            "config/config.json",
            os.path.join(os.path.dirname(__file__), "config.json"),
            os.path.join(os.path.dirname(__file__), "..", "config.json")
        ]
        
        config_path = None
        for path in config_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        _default_config_manager = ConfigManager(config_path)
    
    return _default_config_manager.get_config()


def init_config(config_path: Optional[str] = None) -> ConfigManager:
    """Initialize configuration manager"""
    global _default_config_manager
    _default_config_manager = ConfigManager(config_path)
    return _default_config_manager


# Example configuration file template
EXAMPLE_CONFIG = {
    "llm": {
        "provider": "azure_openai",
        "model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 2000,
        "timeout": 30
    },
    "rag": {
        "enabled": True,
        "vector_db_path": "data/vector_db",
        "similarity_threshold": 0.7,
        "max_results": 5
    },
    "image_analysis": {
        "enabled": True,
        "provider": "azure_computer_vision",
        "max_image_size_mb": 10
    },
    "github": {
        "api_base_url": "https://api.github.com",
        "timeout": 30,
        "max_retries": 3
    },
    "webhook": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": False
    },
    "processing": {
        "max_concurrent_requests": 10,
        "request_timeout": 300,
        "enable_async_processing": True
    },
    "logging": {
        "level": "INFO",
        "enable_console": True,
        "file_path": "logs/intelligent_dispatcher.log"
    },
    "enable_intent_analysis": True,
    "enable_rag_analysis": True,
    "enable_image_analysis": True,
    "min_issue_title_length": 5,
    "min_issue_body_length": 20
}


if __name__ == "__main__":
    # Create example configuration file
    example_path = "config_example.json"
    
    with open(example_path, 'w', encoding='utf-8') as f:
        json.dump(EXAMPLE_CONFIG, f, indent=2)
    
    print(f"Example configuration file created: {example_path}")
    
    # Test configuration loading
    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        print(f"Configuration loaded successfully")
        print(f"LLM Provider: {config.llm.provider}")
        print(f"Webhook Port: {config.webhook.port}")
        print(f"RAG Enabled: {config.enable_rag_analysis}")
    except Exception as e:
        print(f"Configuration test failed: {str(e)}")
