# Task 1: Project Setup Guide

## Overview
This task establishes the foundation for your MLOps pipeline with proper version control, directory structure, and DVC configuration.

## Step-by-Step Setup

### Step 1: Create Project Directory

```bash
# Create main project folder
mkdir cats-dogs-mlops
cd cats-dogs-mlops
```

### Step 2: Initialize Git Repository

```bash
git init

# Configure Git user (if not done globally)
git config user.name "Your Name"
git config user.email "your.email@extremenetworks.com"

# Add initial files
git add .
git commit -m "Initial project setup"
```

### Step 3: Create Directory Structure

```bash
# Create all required directories
mkdir -p data/raw data/processed
mkdir -p models
mkdir -p tests
mkdir -p config
mkdir -p app
mkdir -p docker
mkdir -p .github/workflows
mkdir -p logs
mkdir -p artifacts
```

### Step 4: Initialize DVC

```bash
# Install DVC (if not in requirements.txt yet)
pip install dvc

# Initialize DVC
dvc init

# Create DVC remote (optional but recommended)
mkdir -p /tmp/dvc-storage  # Local storage
dvc remote add -d storage /tmp/dvc-storage
```

### Step 5: Add Project Files

Copy/create these files in your project root:

```
cats-dogs-mlops/
├── .gitignore          ✓ (provided)
├── .dvcignore          ✓ (provided)
├── README.md           ✓ (provided)
├── requirements.txt    ✓ (provided)
├── config/
│   ├── config.yaml     ✓ (provided)
│   └── dvc.yaml        ✓ (provided)
├── data/
│   ├── raw/            (for raw dataset)
│   └── processed/      (for preprocessed images)
├── models/             (for trained models)
├── src/                (for Python scripts - create in Task 2)
├── tests/              (for unit tests - create in Task 7)
└── app/                (for FastAPI app - create in Task 4)
```

### Step 6: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 7: Create .env File (Optional)

```bash
# Create .env for local development
cat > .env << EOF
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_PATH=models/best_model.h5
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development
EOF
```

### Step 8: Add .env to .gitignore

```bash
# Add to .gitignore if not already there
echo ".env" >> .gitignore
echo "venv/" >> .gitignore
```

### Step 9: Initial Git Commit

```bash
# Stage all files
git add .

# Create initial commit
git commit -m "feat: Initialize MLOps project structure with Git and DVC"

# View commit log
git log --oneline
```

### Step 10: Verify DVC Setup

```bash
# Check DVC status
dvc status

# Check DVC remote
dvc remote list

# View DVC config
cat .dvc/config
```

## File Checklist

- [ ] `.gitignore` - Excludes Python cache, models, data, IDE files
- [ ] `.dvcignore` - DVC ignore rules
- [ ] `README.md` - Project documentation
- [ ] `requirements.txt` - Python dependencies
- [ ] `config/config.yaml` - Main configuration
- [ ] `config/dvc.yaml` - DVC pipeline definition
- [ ] `data/raw/` - Directory for raw dataset
- [ ] `data/processed/` - Directory for preprocessed data
- [ ] `models/` - Directory for trained models
- [ ] `tests/` - Directory for unit tests
- [ ] `src/` - Directory for source code
- [ ] `app/` - Directory for FastAPI app
- [ ] `docker/` - Directory for Dockerfile
- [ ] `.github/workflows/` - Directory for GitHub Actions

## Verification

Run these commands to verify setup:

```bash
# Check Git status
git status
git log

# Check DVC initialization
dvc config -l
dvc remote list

# Verify virtual environment
python --version
pip list | grep tensorflow

# Test imports
python -c "import tensorflow; import fastapi; print('All imports successful!')"
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `command not found: dvc` | Install DVC: `pip install dvc` |
| `No such file or directory: data/raw` | Run: `mkdir -p data/raw data/processed` |
| `fatal: not a git repository` | Run: `git init` in project directory |
| Python version issues | Use Python 3.9+ (`python --version`) |
| Permission denied in venv | Run: `chmod +x venv/bin/activate` |

## Next Steps

Once Task 1 is complete:
1. ✅ Project structure ready
2. ✅ Git initialized with initial commit
3. ✅ DVC configured for data versioning
4. ✅ Python environment set up

**Proceed to Task 2**: Download and prepare Cats vs Dogs dataset

## References

- [Git Documentation](https://git-scm.com/doc)
- [DVC Getting Started](https://dvc.org/doc/start)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
