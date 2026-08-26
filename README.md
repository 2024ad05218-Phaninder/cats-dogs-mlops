# MLOps Pipeline: Cats vs Dogs Classification

End-to-end MLOps project for binary image classification using open-source tools.

## Project Structure

```
cats-dogs-mlops/
├── data/
│   ├── raw/                 # Raw dataset from Kaggle
│   └── processed/           # Preprocessed 224x224 images
├── src/
│   ├── data_preprocessing.py    # Data loading & augmentation
│   ├── model.py                 # CNN model architecture
│   ├── train.py                 # Training script
│   └── inference.py             # Inference utilities
├── models/                  # Trained model artifacts
├── tests/
│   ├── test_preprocessing.py    # Unit tests for data
│   └── test_inference.py        # Unit tests for model
├── config/
│   ├── config.yaml              # Configuration file
│   └── dvc.yaml                 # DVC pipeline config
├── app/
│   └── main.py                  # FastAPI inference service
├── docker/
│   └── Dockerfile               # Container image definition
├── .github/
│   └── workflows/
│       ├── ci.yml               # CI pipeline
│       └── cd.yml               # CD pipeline
├── docker-compose.yml       # Deployment compose file
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup Instructions

### 1. Clone & Initialize

```bash
# Initialize Git repository
git init

# Initialize DVC
dvc init

# Create project directories
mkdir -p data/raw data/processed models tests config app docker .github/workflows logs
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure DVC

```bash
# Set up DVC remote (optional, for centralized storage)
dvc remote add -d myremote /tmp/dvc-storage
dvc config core.autostage true
```

## Modules Overview

- **M1**: Model Development & Experiment Tracking (Tasks 1-3)
- **M2**: Model Packaging & Containerization (Tasks 4-6)
- **M3**: CI Pipeline (Tasks 7-8)
- **M4**: CD Pipeline (Tasks 9-10)
- **M5**: Monitoring & Submission (Tasks 11-12)

## Next Steps

1. Download Cats vs Dogs dataset from Kaggle
2. Track data with DVC: `dvc add data/raw`
3. Build and train the CNN model
4. Create FastAPI inference service
5. Containerize with Docker
6. Set up CI/CD pipelines

## Tools Used

- **Version Control**: Git
- **Data Versioning**: DVC
- **Experiment Tracking**: MLflow
- **Model Framework**: TensorFlow/PyTorch
- **API**: FastAPI
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Testing**: pytest
- **Deployment**: Docker Compose / Kubernetes

## References

- [Cats and Dogs Dataset](https://www.kaggle.com/datasets/shaunqi/catsanddogs)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
