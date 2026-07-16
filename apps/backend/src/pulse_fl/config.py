import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Base paths
    # Since it is inside src/pulse_fl, resolve one extra level up
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    
    # Server Settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # Federated Learning Settings
    MIN_CLIENTS_FOR_AGGREGATION: int = 3
    TOTAL_ROUNDS: int = 10
    GLOBAL_MODELS_DIR: Optional[Path] = None
    CLIENT_UPDATES_DIR: Optional[Path] = None
    
    # Database Connection
    DATABASE_URL: Optional[str] = None
    
    # Ollama Settings
    OLLAMA_API_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "gemma4:latest"
    
    # SMTP Email Alert Settings
    SMTP_HOST: str = "127.0.0.1"
    SMTP_PORT: int = 1025
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SENDER_EMAIL: str = "alerts@pulse-fl.com"
    
    # Model configuration
    ECG_INPUT_CHANNELS: int = 1
    ECG_SEQUENCE_LENGTH: int = 1000  # 2 seconds at 500Hz
    NUM_CLASSES: int = 2             # Normal vs Arrhythmia (binary classification)

    class Config:
        env_prefix = "PULSE_FL_"
        env_file = ".env"
        env_file_encoding = "utf-8"

# Initialize settings
settings = Settings()

# Dynamic defaults if not provided in environment/.env
if not settings.DATABASE_URL:
    settings.DATABASE_URL = f"sqlite:///{settings.STORAGE_DIR}/pulse_fl.db"
if not settings.GLOBAL_MODELS_DIR:
    settings.GLOBAL_MODELS_DIR = settings.STORAGE_DIR / "global_models"
if not settings.CLIENT_UPDATES_DIR:
    settings.CLIENT_UPDATES_DIR = settings.STORAGE_DIR / "client_updates"

# Initialize and verify storage directories
settings.GLOBAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.CLIENT_UPDATES_DIR.mkdir(parents=True, exist_ok=True)
