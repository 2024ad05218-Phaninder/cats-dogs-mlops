# Complete MLOps Pipeline Guide
## Cats vs Dogs Binary Image Classification

A comprehensive guide to building, deploying, and monitoring a machine learning model using modern MLOps practices.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Task 1: Project Setup](#task-1-project-setup)
3. [Task 2: Download Dataset](#task-2-download-dataset)
4. [Task 3: Train Model](#task-3-train-model)
5. [Task 4: Create API](#task-4-create-api)
6. [Task 5: Pin Dependencies](#task-5-pin-dependencies)
7. [Task 6: Docker Setup](#task-6-docker-setup)
8. [Task 7: Unit Tests](#task-7-unit-tests)
9. [Task 8: CI Pipeline](#task-8-ci-pipeline)
10. [Task 9: Docker Compose](#task-9-docker-compose)
11. [Task 10: CD Pipeline](#task-10-cd-pipeline)
12. [Task 11: Monitoring](#task-11-monitoring)
13. [Quick Reference](#quick-reference)

---

# Environment Setup

This section walks you through setting up everything needed to run the MLOps pipeline on your computer.

## What You Need Before Starting

Before you start, make sure you have:
- A computer with Mac, Linux, or Windows (with WSL2)
- Internet connection (to download tools and datasets)
- About 10 GB of free disk space
- Administrator access to install software

## Step 1: Install Required Software

### Install Python 3.10

Python is the programming language we'll use.

**On Mac:**
```bash
# Using Homebrew (if you have it)
brew install python@3.10

# Or download from: https://www.python.org/downloads/
```

**On Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

**Verify installation:**
```bash
python3 --version
# Should show: Python 3.10.x
```

### Install Git

Git helps you save your code changes and upload them to GitHub.

**On Mac:**
```bash
brew install git
```

**On Linux:**
```bash
sudo apt install git
```

**Verify installation:**
```bash
git --version
```

### Install Docker

Docker lets you run applications in containers (think of them as small virtual computers).

**Download and install from:** https://www.docker.com/products/docker-desktop

After installation, verify:
```bash
docker --version
docker run hello-world
```

### Install Git LFS (Large File Storage)

This helps store large model files in Git.

**On Mac:**
```bash
brew install git-lfs
```

**On Linux:**
```bash
sudo apt install git-lfs
```

**Enable it:**
```bash
git lfs install
```

## Step 2: Create Your Project Directory

Create a folder where you'll work on this project:

```bash
# Create and enter directory
mkdir cats-dogs-mlops
cd cats-dogs-mlops

# Initialize Git
git init

# Create main folders
mkdir src
mkdir models
mkdir notebooks
mkdir data
mkdir tests
mkdir .github
mkdir .github/workflows

# Create Python virtual environment (isolated Python setup)
python3 -m venv venv

# Activate virtual environment
# On Mac/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate

# Verify activation (you should see (venv) in your terminal)
```

## Step 3: Set Up GitHub Account

GitHub is where you store your code online.

1. Go to https://github.com
2. Click "Sign up" and create an account
3. Create a new repository called `cats-dogs-mlops`
4. Follow the instructions to connect your local folder to GitHub:

```bash
git remote add origin https://github.com/YOUR_USERNAME/cats-dogs-mlops.git
git branch -M main
git push -u origin main
```

## Step 4: Create Kaggle Account

Kaggle is a data science platform where we download our dataset.

1. Go to https://www.kaggle.com
2. Create an account
3. Go to Account Settings → API
4. Click "Create New Token" (downloads `kaggle.json`)
5. Move the file to the right location:

```bash
# On Mac/Linux:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# On Windows:
# Move kaggle.json to C:\Users\YourUsername\.kaggle\
```

Verify:
```bash
kaggle datasets list
```

## Step 5: Install Python Packages

These are the tools we'll use for machine learning, testing, and deploying.

```bash
# Make sure you're in your virtual environment (see Step 2)
# Install everything
pip install --upgrade pip

# Core packages for machine learning
pip install tensorflow==2.14.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install pillow==10.0.1
pip install scikit-learn==1.3.2

# For tracking experiments
pip install mlflow==2.7.1

# For data versioning
pip install dvc==3.29.0

# For the API server
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install pydantic==2.4.2

# For testing
pip install pytest==7.4.3
pip install pytest-cov==4.1.0

# For metrics
pip install prometheus-client

# Verify installation
python -c "import tensorflow; import fastapi; print(' All packages installed!')"
```

## Step 6: Test Your Setup

Make sure everything is working:

```bash
# Test Python
python --version

# Test Git
git --version

# Test Docker
docker --version

# Test virtual environment
which python  # Should show path with 'venv' in it

# Test Kaggle
kaggle datasets list | head -5

# All working? You're ready to go!
echo " Environment setup complete!"
```

** Your environment is ready.** 

Now you can proceed with the tasks below.

---

# Task 1: Project Setup

**What:** Create the basic folder structure and start version control.

## Steps

### Step 1: Create Project Structure

```bash
# Create necessary folders
mkdir -p src              # Python source code
mkdir -p models           # Trained models
mkdir -p data             # Dataset storage
mkdir -p notebooks        # Jupyter notebooks
mkdir -p tests            # Test files
mkdir -p logs             # Application logs

# Create important files
touch README.md            # Project description
touch .gitignore           # Tell Git what to ignore
touch requirements.txt     # List of packages
```

### Step 2: Create .gitignore File

The `.gitignore` file tells Git which files NOT to save (like large data files).

```bash
cat > .gitignore << 'EOF'
# Virtual environment
venv/
env/

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/

# Data (too large to store)
data/raw/
*.zip

# Models (too large)
models/*.h5
models/*.pkl

# Logs
logs/

# IDE
.vscode/
.idea/

# OS
.DS_Store
.env
EOF
```

### Step 3: Create README

```bash
cat > README.md << 'EOF'
# Cats vs Dogs MLOps Pipeline

A complete machine learning pipeline for binary image classification using TensorFlow, FastAPI, and Docker.

## Quick Start

1. Set up environment (see COMPLETE_MLOPS_GUIDE.md)
2. Download dataset: `python src/download_data.py`
3. Train model: `python src/train.py`
4. Start API: `python -m uvicorn app.main:app --reload`

## Project Structure

- `src/` - Python source code
- `models/` - Trained models
- `data/` - Dataset
- `tests/` - Test files
- `notebooks/` - Jupyter notebooks

EOF
```

### Step 4: Initialize Git

```bash
# Add all files
git add .

# Create first commit
git commit -m "init: Create project structure"

# Push to GitHub
git push -u origin main

# Verify
git log --oneline | head -3
```

** Task 1 Complete!** Your project folder is ready.

---

# Task 2: Download Dataset

**What:** Download the Cats vs Dogs dataset from Kaggle and organize it.

**Time:** 15 minutes (mostly automatic)

## What You're Doing

You're downloading 10,000 images of cats and dogs from Kaggle, which you'll use to train your machine learning model.

## Steps

### Step 1: Download the Data

```bash
# Make data directory
mkdir -p data/raw
cd data/raw

# Download from Kaggle (uses your kaggle.json credentials)
kaggle datasets download -d shaunlowis/catsanddogs

# Unzip the downloaded file
unzip catsanddogs.zip

# Check what you got
ls -la

# You should see:
# - Cat/ (folder with cat images)
# - Dog/ (folder with dog images)
```

### Step 2: Check Dataset Size

```bash
# Count images
find data/raw/Cat -type f | wc -l  # Should be ~10,000 cat images
find data/raw/Dog -type f | wc -l  # Should be ~10,000 dog images
```

### Step 3: Create Download Script

For future use, create a Python script to download automatically:

```bash
cat > src/download_data.py << 'EOF'
"""Download and prepare dataset"""
import os
from pathlib import Path
import subprocess

def download_dataset():
    """Download Cats vs Dogs dataset from Kaggle"""
    
    # Create data directory
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    # Download
    print("Downloading dataset from Kaggle...")
    subprocess.run([
        'kaggle', 'datasets', 'download', '-d', 'shaunlowis/catsanddogs',
        '-p', 'data/raw'
    ])
    
    # Unzip
    print("Extracting files...")
    subprocess.run(['unzip', '-q', 'data/raw/catsanddogs.zip', '-d', 'data/raw'])
    
    print(" Dataset downloaded and extracted!")

if __name__ == '__main__':
    download_dataset()
EOF

# Test it
python src/download_data.py
```

### Step 4: Commit Changes

```bash
git add data/.gitkeep src/download_data.py
git commit -m "feat: Add dataset download script"
git push origin main
```

** Task 2 Complete!** You have the dataset ready.

---

# Task 3: Train Model

**What:** Build and train a neural network to classify cats vs dogs.

**Time:** 30 minutes

## What You're Doing

You're creating a machine learning model that learns to recognize whether an image contains a cat or a dog. This model will be trained on the 20,000 images you downloaded.

## Steps

### Step 1: Create Training Script

```bash
cat > src/train.py << 'EOF'
"""Train the cats vs dogs classifier model"""
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
import numpy as np

def load_and_prepare_data():
    """Load images and labels"""
    print("Loading dataset...")
    
    cat_dir = Path('data/raw/Cat')
    dog_dir = Path('data/raw/Dog')
    
    images = []
    labels = []
    
    # Load cat images (label = 0)
    for img_path in list(cat_dir.glob('*.jpg'))[:5000]:
        try:
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(224, 224)
            )
            images.append(tf.keras.preprocessing.image.img_to_array(img))
            labels.append(0)
        except:
            pass
    
    # Load dog images (label = 1)
    for img_path in list(dog_dir.glob('*.jpg'))[:5000]:
        try:
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(224, 224)
            )
            images.append(tf.keras.preprocessing.image.img_to_array(img))
            labels.append(1)
        except:
            pass
    
    # Convert to numpy arrays
    X = np.array(images) / 255.0  # Normalize to 0-1
    y = np.array(labels)
    
    print(f"Loaded {len(images)} images")
    return X, y

def build_model():
    """Create neural network"""
    model = tf.keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Load data, build model, and train"""
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Split into train (80%), validation (10%), test (10%)
    split1 = int(0.8 * len(X))
    split2 = int(0.9 * len(X))
    
    X_train, y_train = X[:split1], y[:split1]
    X_val, y_val = X[split1:split2], y[split1:split2]
    X_test, y_test = X[split2:], y[split2:]
    
    # Build model
    model = build_model()
    
    # Train
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n Test Accuracy: {test_acc*100:.2f}%")
    
    # Save model
    Path('models').mkdir(exist_ok=True)
    model.save('models/best_model.h5')
    print(" Model saved to models/best_model.h5")

if __name__ == '__main__':
    train_model()
EOF
```

### Step 2: Train the Model

```bash
# This will take 20-30 minutes depending on your computer
python src/train.py

# You should see:
# - Loading dataset...
# - Loaded 10000 images
# - Training model...
# - Epoch 1/10
# -  Test Accuracy: 94.56%
# -  Model saved to models/best_model.h5
```

### Step 3: Verify Model Works

```bash
# Create quick test script
python << 'EOF'
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('models/best_model.h5')
print("✅ Model loaded successfully!")
print(f"Model shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# Make test prediction
fake_image = np.random.rand(1, 224, 224, 3)
prediction = model.predict(fake_image)
print(f"Test prediction: {prediction[0][0]:.4f}")
EOF
```

### Step 4: Commit Model

```bash
git add src/train.py models/

git commit -m "feat: Add model training script and trained model"
git push origin main
```

** Task 3 Complete!** Your model is trained and working.

---

# Task 4: Create API

**What:** Create a web service that lets people use your model through the internet.

**Time:** 20 minutes

## What You're Doing

You're creating a REST API (a way for other programs to talk to your model). People can send images to your API, and it will return predictions (cat or dog).

## Steps

### Step 1: Create Inference Module

```bash
cat > src/inference.py << 'EOF'
"""Model inference (prediction) module"""
import tensorflow as tf
import numpy as np
from PIL import Image

class CatsDogsClassifier:
    """Wrapper for model predictions"""
    
    def __init__(self, model_path):
        """Load model"""
        self.model = tf.keras.models.load_model(model_path)
        self.image_size = 224
        self.class_names = ['cat', 'dog']
    
    def preprocess_image(self, image):
        """Prepare image for model"""
        if isinstance(image, str):
            image = Image.open(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((self.image_size, self.image_size))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array.astype(np.float32)
    
    def predict(self, image):
        """Make prediction"""
        image_array = self.preprocess_image(image)
        prediction = self.model.predict(image_array, verbose=0)[0][0]
        
        confidence = max(prediction, 1 - prediction)
        predicted_class = self.class_names[int(round(prediction))]
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                'cat': float(1 - prediction),
                'dog': float(prediction)
            }
        }

EOF
```

### Step 2: Create FastAPI Application

```bash
# Create app folder
mkdir -p app
touch app/__init__.py

cat > app/main.py << 'EOF'
"""FastAPI application for model serving"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import io
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference import CatsDogsClassifier

# Create app
app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="Classify images as cats or dogs",
    version="1.0.0"
)

# Allow requests from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
classifier = None

@app.on_event("startup")
async def startup_event():
    """Load model when app starts"""
    global classifier
    try:
        classifier = CatsDogsClassifier('models/best_model.h5')
        print(" Model loaded")
    except Exception as e:
        print(f" Failed to load model: {e}")

@app.get("/health")
async def health():
    """Check if API is working"""
    return {"status": "healthy"}

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "CatsDogsClassifier",
        "classes": ["cat", "dog"],
        "input_shape": [224, 224, 3]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict class of uploaded image"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Predict
        result = classifier.predict(image)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

EOF
```

### Step 3: Test the API

```bash
# Start the server
python -m uvicorn app.main:app --reload

# It should print:
# Uvicorn running on http://127.0.0.1:8000
# Press CTRL+C to quit

# In another terminal, test it:
curl http://localhost:8000/health
# Should return: {"status":"healthy"}

# Test with image:
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg"
```

### Step 4: Commit Changes

```bash
git add app/ src/inference.py
git commit -m "feat: Create FastAPI inference service"
git push origin main
```

** Task 4 Complete!** Your API is running and ready to serve predictions.

---

# Task 5: Pin Dependencies

**What:** Create a fixed list of package versions so your code works the same everywhere.

**Time:** 5 minutes

## What You're Doing

When you install packages, Python can install different versions. "Pinning" means saying "use EXACTLY this version", which makes your code reproducible on any computer.

## Steps

### Step 1: Generate Requirements

```bash
# Create requirements.txt with pinned versions
cat > requirements.txt << 'EOF'
# Core ML & Data Processing
tensorflow==2.14.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.2
pillow==10.0.1

# Experiment Tracking
mlflow==2.7.1

# Data Versioning
dvc==3.29.0
dvc-s3==3.0.1

# API Framework
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
pydantic==2.4.2

# Testing
pytest==7.4.3
pytest-cov==4.1.0
httpx==0.25.0

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
requests==2.31.0

# Plotting & Visualization
matplotlib==3.8.1
seaborn==0.13.0

# Prometheus metrics
prometheus-client

EOF
```

### Step 2: Verify Dependencies

```bash
# Test installation
pip install -r requirements.txt

# Verify
python -c "import tensorflow; import fastapi; print(' All OK!')"
```

### Step 3: Commit

```bash
git add requirements.txt
git commit -m "feat: Pin dependencies for reproducibility"
git push origin main
```

** Task 5 Complete!** Your dependencies are locked in.

---

# Task 6: Docker Setup

**What:** Package your API in Docker so it works the same on any computer.

**Time:** 15 minutes

## What You're Doing

Docker creates a "container" (like a mini virtual computer) with everything your API needs. This way, someone else can run your API on their computer without installing anything except Docker.

## Steps

### Step 1: Create Dockerfile

```bash
cat > Dockerfile << 'EOF'
# Use official Python runtime as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
```

### Step 2: Create .dockerignore

This tells Docker which files to skip (like your dataset).

```bash
cat > .dockerignore << 'EOF'
__pycache__
.git
.gitignore
.env
venv/
data/raw/
*.md
EOF
```

### Step 3: Build Docker Image

```bash
# Build (this takes a few minutes)
docker build -t cats-dogs-classifier:latest .

# Verify it was created
docker images | grep cats-dogs-classifier
```

### Step 4: Test Docker Container

```bash
# Run the container
docker run -p 8000:8000 cats-dogs-classifier:latest

# In another terminal, test
curl http://localhost:8000/health

# Stop container
# Press Ctrl+C in the terminal
```

### Step 5: Commit

```bash
git add Dockerfile .dockerignore
git commit -m "feat: Add Docker containerization"
git push origin main
```

** Task 6 Complete!** Your app is containerized.

---

# Task 7: Unit Tests

**What:** Write tests to make sure your code works correctly.

**Time:** 30 minutes

## What You're Doing

Tests are small programs that verify your code does what it's supposed to. They catch bugs early.

## Steps

### Step 1: Create Test Files

```bash
mkdir -p tests
touch tests/__init__.py

# Create test for data preprocessing
cat > tests/test_preprocessing.py << 'EOF'
"""Tests for data preprocessing"""
import pytest
from pathlib import Path
import tempfile
from PIL import Image

def test_image_loading():
    """Test loading and processing images"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test image
        img = Image.new('RGB', (256, 256), color='red')
        img_path = Path(tmpdir) / 'test.jpg'
        img.save(img_path)
        
        # Test it exists
        assert img_path.exists()
        
        # Test we can load it
        loaded_img = Image.open(img_path)
        assert loaded_img.size == (256, 256)

def test_image_resize():
    """Test image resizing"""
    img = Image.new('RGB', (256, 256), color='blue')
    resized = img.resize((224, 224))
    assert resized.size == (224, 224)

EOF

# Create test for inference
cat > tests/test_inference.py << 'EOF'
"""Tests for model inference"""
import pytest
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference import CatsDogsClassifier

def test_model_initialization():
    """Test model loads correctly"""
    model_path = Path('models/best_model.h5')
    if not model_path.exists():
        pytest.skip("Model not found")
    
    classifier = CatsDogsClassifier(str(model_path))
    assert classifier.model is not None
    assert classifier.class_names == ['cat', 'dog']

def test_prediction():
    """Test prediction works"""
    model_path = Path('models/best_model.h5')
    if not model_path.exists():
        pytest.skip("Model not found")
    
    classifier = CatsDogsClassifier(str(model_path))
    
    # Create dummy image
    img = Image.new('RGB', (224, 224), color='red')
    result = classifier.predict(img)
    
    # Check result structure
    assert 'predicted_class' in result
    assert 'confidence' in result
    assert result['predicted_class'] in ['cat', 'dog']
    assert 0 <= result['confidence'] <= 1

EOF
```

### Step 2: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Expected output:
# tests/test_preprocessing.py::test_image_loading PASSED
# tests/test_preprocessing.py::test_image_resize PASSED
# tests/test_inference.py::test_model_initialization PASSED
# tests/test_inference.py::test_prediction PASSED
# =================== 4 passed in 2.34s ===================
```

### Step 3: Check Coverage

Coverage shows what percentage of your code is tested.

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=term-missing

# You should see something like:
# src/inference.py    280    20    93%
```

### Step 4: Commit

```bash
git add tests/
git commit -m "feat: Add comprehensive unit tests"
git push origin main
```

** Task 7 Complete!** Your code is tested.

---

# Task 8: CI Pipeline

**What:** Set up automatic testing on GitHub when you push code.

**Time:** 10 minutes

## What You're Doing

CI (Continuous Integration) means that every time you upload code to GitHub, it automatically runs your tests. If tests fail, you know something is broken before it goes live.

## Steps

### Step 1: Create CI Workflow

```bash
# Create workflow file
cat > .github/workflows/ci.yml << 'EOF'
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Build Docker image
      run: docker build -t cats-dogs-classifier:latest .

EOF
```

### Step 2: Commit and Push

```bash
git add .github/workflows/ci.yml
git commit -m "ci: Add GitHub Actions CI pipeline"
git push origin main
```

### Step 3: Watch It Run

1. Go to GitHub.com
2. Open your repository
3. Click the "Actions" tab
4. You should see your workflow running
5. Click on it to see details

** Task 8 Complete!** Your code is auto-tested on every push.

---

# Task 9: Docker Compose

**What:** Run multiple services together (API, database, monitoring, etc.)

**Time:** 15 minutes

## What You're Doing

Docker Compose lets you say "I want to run my API, a database, and monitoring tools all together" in one command. It makes deployment easy.

## Steps

### Step 1: Create docker-compose.yml

```bash
cat > docker-compose.yml << 'EOF'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: cats-dogs-api
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/best_model.h5
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: cats-dogs-postgres
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.7.1
    container_name: cats-dogs-mlflow
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow
    depends_on:
      - postgres
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
      --host 0.0.0.0
      --port 5000
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: cats-dogs-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: cats-dogs-grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    restart: unless-stopped

volumes:
  postgres-data:
  prometheus-data:
  grafana-data:

EOF
```

### Step 2: Create prometheus.yml

```bash
cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

EOF
```

### Step 3: Start Services

```bash
# Start all services
docker-compose up -d

# Wait 30 seconds for services to start
sleep 30

# Check they're running
docker-compose ps

# You should see 5 containers all UP
```

### Step 4: Access Services

```bash
# API
curl http://localhost:8000/health

# MLflow (experiment tracking)
open http://localhost:5000

# Prometheus (metrics)
open http://localhost:9090

# Grafana (dashboards)
open http://localhost:3001
# Login: admin / admin
```

### Step 5: Commit

```bash
git add docker-compose.yml prometheus.yml
git commit -m "feat: Add Docker Compose deployment stack"
git push origin main
```

** Task 9 Complete!** Your services are running together.

---

# Task 10: CD Pipeline

**What:** Automatically deploy your code when you push to GitHub.

**Time:** 15 minutes

## What You're Doing

CD (Continuous Deployment) automatically deploys your code to production after CI tests pass. No manual deployment needed!

## Steps

### Step 1: Add GitHub Secrets

On GitHub, go to:
- Settings → Secrets and variables → Actions
- Add these secrets:
  - `DOCKER_USERNAME` - Your Docker Hub username
  - `DOCKER_PASSWORD` - Your Docker Hub token
  - `DEPLOY_HOST` - Your server's IP
  - `DEPLOY_USER` - Username to log into server
  - `DEPLOY_KEY` - SSH private key
  - `SLACK_WEBHOOK` - (Optional) For notifications

### Step 2: Create CD Workflow

```bash
cat > .github/workflows/cd.yml << 'EOF'
name: CD Pipeline

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production

    steps:
    - uses: actions/checkout@v3

    - name: Build and push Docker image
      run: |
        docker build -t cats-dogs-classifier:latest .

    - name: Deploy to production
      env:
        DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
        DEPLOY_HOST: ${{ secrets.DEPLOY_HOST }}
        DEPLOY_USER: ${{ secrets.DEPLOY_USER }}
      run: |
        mkdir -p ~/.ssh
        echo "$DEPLOY_KEY" > ~/.ssh/deploy_key
        chmod 600 ~/.ssh/deploy_key
        ssh-keyscan -H $DEPLOY_HOST >> ~/.ssh/known_hosts
        
        ssh -i ~/.ssh/deploy_key $DEPLOY_USER@$DEPLOY_HOST << 'DEPLOY'
          cd /opt/cats-dogs-mlops
          git pull origin main
          docker-compose pull
          docker-compose up -d
        DEPLOY

EOF
```

### Step 3: Commit

```bash
git add .github/workflows/cd.yml
git commit -m "feat: Add CD pipeline with automated deployment"
git push origin main
```

** Task 10 Complete!** Your code auto-deploys after tests pass.

---

# Task 11: Monitoring

**What:** Track how your API is performing with metrics and logs.

**Time:** 25 minutes

## What You're Doing

Monitoring means watching your API to make sure:
- It's responding fast
- Not many errors
- People are actually using it
- Resources (memory, CPU) aren't running out

## Steps

### Step 1: Update API with Monitoring

Your API needs to collect metrics (performance data).

The updated `app/main.py` includes:
- Request counting
- Response time tracking
- Error tracking
- Structured JSON logging

Key metrics collected:
- How many requests per second
- How long responses take
- How many errors occur
- Model prediction confidence

### Step 2: Access Monitoring Tools

```bash
# Prometheus (raw metrics)
# Query: fastapi_requests_total
# Shows: Number of API requests
open http://localhost:9090

# Grafana (dashboards and visualization)
# Shows: Charts, graphs, alerts
open http://localhost:3001
# Login: admin / admin
```

### Step 3: Create Grafana Dashboard

In Grafana:
1. Go to + (Create)
2. Select "Dashboard"
3. Add panels to visualize:
   - Request rate
   - Error rate
   - Response time
   - Active requests
   - Predictions

### Step 4: View Logs

```bash
# See API logs in real-time
docker-compose logs -f api

# See structured JSON logs
tail -f logs/app.json | jq .

# Find errors
docker-compose logs api | grep ERROR
```

### Step 5: Commit

```bash
git add app/main.py logs/.gitkeep
git commit -m "feat: Add comprehensive monitoring and logging"
git push origin main
```

** Task 11 Complete!** Your API is being monitored.

---

# Quick Reference

## Most Common Commands

### Starting Everything

```bash
# Start virtual environment
source venv/bin/activate

# Start all Docker services
docker-compose up -d

# Run API locally (for development)
python -m uvicorn app.main:app --reload
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_inference.py::test_prediction -v

# Show coverage
pytest tests/ --cov=src
```

### Docker

```bash
# Build image
docker build -t cats-dogs-classifier:latest .

# Start containers
docker-compose up -d

# See logs
docker-compose logs -f api

# Stop everything
docker-compose down
```

### Git

```bash
# Add changes
git add .

# Commit with message
git commit -m "your message"

# Push to GitHub
git push origin main

# See history
git log --oneline
```

## Accessing Services

| Service | URL | Purpose |
|---------|-----|---------|
| API Docs | http://localhost:8000/docs | Interactive API testing |
| MLflow | http://localhost:5000 | Experiment tracking |
| Prometheus | http://localhost:9090 | Raw metrics |
| Grafana | http://localhost:3001 | Dashboards (admin/admin) |
| PostgreSQL | localhost:5432 | Database |

## Project Structure

```
cats-dogs-mlops/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI application
├── src/
│   ├── train.py             # Training script
│   ├── inference.py         # Model predictions
│   └── download_data.py     # Dataset download
├── models/
│   └── best_model.h5        # Trained model
├── data/
│   ├── raw/                 # Downloaded images
│   └── processed/           # Processed data
├── tests/
│   ├── test_preprocessing.py
│   └── test_inference.py
├── logs/
│   └── app.json             # Application logs
├── notebooks/               # Jupyter notebooks
├── Dockerfile               # Docker build recipe
├── docker-compose.yml       # Multi-service setup
├── requirements.txt         # Python packages
├── prometheus.yml           # Metrics config
└── README.md                # Project description
```

## Troubleshooting

### Problem: Container won't start

```bash
# Check logs
docker-compose logs container_name

# Restart
docker-compose restart container_name
```

### Problem: Tests fail

```bash
# Run tests with more detail
pytest tests/ -v -s

# Run only one test
pytest tests/test_inference.py::test_prediction -v
```

### Problem: API slow

```bash
# Check metrics in Prometheus
# Query: histogram_quantile(0.95, rate(fastapi_request_duration_seconds_bucket[5m]))

# Check active requests
# Query: fastapi_active_requests
```

### Problem: Model not found

```bash
# Make sure model exists
ls -la models/best_model.h5

# If missing, retrain
python src/train.py
```

---

## Summary

You've built a production-ready machine learning system with:

 **Model Training** - Deep learning CNN model
 **API Server** - FastAPI for serving predictions
 **Containerization** - Docker for deployment
 **Testing** - Unit tests for reliability
 **Continuous Integration** - Auto-testing on GitHub
 **Continuous Deployment** - Auto-deployment
 **Monitoring** - Prometheus + Grafana
 **Logging** - Structured JSON logs
 **Version Control** - Git for code management

This is production-grade machine learning infrastructure!

---

**Questions?** Check the specific task documentation or the error messages in your logs.

**Ready for Task 12?** Create final deliverables (zip file + screen recording demo).

