# models.py
import os
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, validator

class ConfigModel(BaseModel):
    """Enhanced configuration model with Pydantic validation"""
    OPENAI_API_KEY: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API Key"
    )
    TELEGRAM_BOT_TOKEN: str = Field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""),
        description="Telegram Bot Token"
    )
    MAX_TOKENS: int = Field(default=300, ge=50, le=1000, 
                             description="Maximum tokens for content generation")
    MODEL_NAME: str = Field(default="gpt-3.5-turbo", 
                             description="OpenAI model to use")

    model_config = ConfigDict(
        validate_assignment=True,
        extra='ignore'
    )

    @validator('OPENAI_API_KEY', 'TELEGRAM_BOT_TOKEN')
    def check_not_empty(cls, v):
        """Ensure API keys are not empty"""
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        return v

class PlatformContent(BaseModel):
    """Model to represent content for different platforms"""
    twitter: Optional[str] = None
    facebook: Optional[str] = None
    linkedin: Optional[str] = None
    instagram: Optional[str] = None
    youtube: Optional[str] = None
    youtube_shorts: Optional[str] = None
    tiktok: Optional[str] = None
    facebook_reels: Optional[str] = None
    research_context: Optional[Dict[str, Any]] = None

class ContentGenerationRequest(BaseModel):
    """Model to validate content generation requests"""
    topic: str = Field(
        ...,  # Required field
        min_length=3, 
        max_length=200,
        description="Topic for content generation"
    )

def validate_config():
    """Load and validate configuration"""
    try:
        config = ConfigModel()
        return config
    except Exception as e:
        print(f"Configuration Validation Error: {e}")
        raise

from typing import List, Literal, Dict
from pydantic import BaseModel, Field, validator

class MediaFile(BaseModel):
    """Model to represent media files with validation"""
    file_path: str
    file_type: Literal['image', 'video']
    platform_type: Literal['jpg', 'png', 'gif']

    @validator('platform_type')
    def validate_file_type(cls, v, values):
        """Validate file types based on platform restrictions"""
        platform_restrictions = {
            'facebook': ['jpg', 'png', 'gif'],
            'instagram': ['jpg', 'png'],
            'twitter': ['jpg', 'png', 'gif'],
            'linkedin': ['jpg', 'png', 'gif']
        }

        # Validate file type
        if 'platform_type' in values:
            platform = values.get('platform_type')
            if v not in platform_restrictions.get(platform, []):
                raise ValueError(f"Unsupported file type for {platform}")
        
        return v

class MediaConfig(BaseModel):
    """Configuration for media upload limits"""
    max_images: Dict[str, int] = {
        'facebook': 10,
        'instagram': 10,
        'twitter': 4,
        'linkedin': 9
    }
    max_videos: Dict[str, int] = {
        'facebook': 1,
        'instagram': 1,
        'linkedin': 1,
        'twitter': 1
    }