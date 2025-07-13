import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'crypto_predictor_2024_secret_key'
    DATABASE_FILE = os.environ.get('DATABASE_FILE') or 'crypto_data.json'

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 