#!/usr/bin/env python3
"""
Colab setup script for the multimodal fact-checking project.
Safely installs dependencies without conflicts with Colab's pre-installed packages.
"""

import subprocess
import sys
import os

def run_command(command, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    if result.stdout:
        print(result.stdout)
    return True

def setup_colab():
    """Setup the project in Colab environment."""
    print("🚀 Setting up multimodal fact-checking project in Colab...")
    
    # 1. Clone repository if not already present
    if not os.path.exists("multimodal-fact-checking"):
        print("📥 Cloning repository...")
        run_command("git clone https://github.com/Fabian2321/multimodal-fact-checking.git")
    
    # 2. Change to project directory
    os.chdir("multimodal-fact-checking")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # 3. Install only essential packages that don't conflict
    print("📦 Installing essential packages...")
    
    # Install core packages one by one to avoid conflicts
    packages = [
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4", 
        "transformers>=4.30.0",
        "tqdm>=4.65.0",
        "requests>=2.28.0",
        "Pillow>=9.5.0",
        "python-json-logger>=2.0.0"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        success = run_command(f"pip install {package}", check=False)
        if not success:
            print(f"⚠️  Warning: Could not install {package}")
    
    # 4. Check GPU availability
    print("🔍 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("ℹ️  GPU not available, using CPU")
    except ImportError:
        print("⚠️  PyTorch not available")
    
    # 5. Test imports
    print("🧪 Testing imports...")
    try:
        import torch
        import transformers
        import sentence_transformers
        import faiss
        import numpy as np
        import pandas as pd
        print("✅ All essential packages imported successfully!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # 6. Create data directories
    print("📁 Creating data directories...")
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/downloaded_fakeddit_images", exist_ok=True)
    os.makedirs("data/external_knowledge", exist_ok=True)
    os.makedirs("data/knowledge_base", exist_ok=True)
    os.makedirs("data/optimization_results", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("✅ Setup complete! You can now run your experiments.")
    print("\n📋 Next steps:")
    print("1. Upload your Fakeddit CSV files to data/raw/")
    print("2. Run: python src/pipeline.py --help")
    print("3. Start with a simple experiment")
    
    return True

if __name__ == "__main__":
    setup_colab() 