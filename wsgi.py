#!/usr/bin/env python3
"""
WSGI entry point para despliegue en producción
"""

import os
from app import app

if __name__ == "__main__":
    app.run() 