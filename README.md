# 🔮 Customer Churn Prediction - End-to-End ML Project

A comprehensive machine learning solution for predicting customer churn in the telecommunications industry, featuring a complete MLOps pipeline from data preprocessing to production deployment on AWS.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Screenshot
![WhatsApp Image 2025-10-26 at 04 49 32](https://github.com/user-attachments/assets/e217a743-160b-4a3b-9189-59d4cea3f3f1)



## 🎯 Overview

### Purpose

Build and ship a production-ready machine learning solution for predicting customer churn in telecom settings—from data preparation and modeling to a REST API and web UI deployed on AWS.

### Problem Solved & Benefits

- **Faster Decisions**: Predicts which customers are likely to churn so teams can act before they leave
- **Operationalized ML**: Model is accessible via REST API and user-friendly web interface
- **Repeatable Delivery**: CI/CD + containers ensure consistent rebuilds, testing, and deployments
- **Traceable Experiments**: MLflow tracks runs, metrics, and artifacts for reproducibility and auditing
- **Production Ready**: Comprehensive data validation, feature engineering, and model serving pipeline

### Key Technologies

- **ML Framework**: XGBoost with scikit-learn
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI with Pydantic validation
- **Web UI**: Gradio for interactive predictions
- **Data Validation**: Great Expectations
- **Containerization**: Docker
- **Cloud Platform**: AWS (ECS Fargate, ALB, CloudWatch)
- **CI/CD**: GitHub Actions

## 🏗️ Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  ML Pipeline    │───▶│  Model Serving  │
│                 │    │                 │    │                 │
│ • Raw CSV Data  │    │ • Data Loading  │    │ • FastAPI API   │
│ • Customer Data │    │ • Validation    │    │ • Gradio UI     │
└─────────────────┘    │ • Preprocessing │    │ • Docker        │
                       │ • Feature Eng.  │    └─────────────────┘
                       │ • Model Training│             │
                       │ • MLflow Logging│             ▼
                       └─────────────────┘    ┌─────────────────┐
                                              │   AWS Cloud     │
                                              │                 │
                                              │ • ECS Fargate   │
                                              │ • ALB           │
                                              │ • CloudWatch    │
                                              └─────────────────┘
```

### ML Pipeline Components

1. **Data Pipeline** (`src/data/`)
   - Data loading with error handling
   - Comprehensive data validation using Great Expectations
   - Data preprocessing and cleaning

2. **Feature Engineering** (`src/features/`)
   - Binary encoding for categorical features
   - One-hot encoding for multi-category features
   - Feature consistency between training and serving

3. **Model Training** (`src/models/`)
   - XGBoost classifier with hyperparameter tuning
   - Model evaluation and metrics tracking
   - MLflow experiment logging

4. **Model Serving** (`src/serving/`)
   - Production inference pipeline
   - Feature transformation consistency
   - Model loading and prediction

5. **Web Application** (`src/app/`)
   - FastAPI REST API with automatic documentation
   - Gradio web interface for interactive predictions
   - Health check endpoints for monitoring

## ✨ Features

### Core ML Features

- **Advanced Feature Engineering**: Binary and one-hot encoding with deterministic mappings
- **Data Quality Validation**: Comprehensive checks using Great Expectations
- **Model Performance**: XGBoost classifier with optimized hyperparameters
- **Experiment Tracking**: Complete MLflow integration for reproducibility
- **Train/Serve Consistency**: Identical feature transformations in training and serving

### API Features

- **RESTful API**: FastAPI with automatic OpenAPI documentation
- **Data Validation**: Pydantic models for request validation
- **Health Monitoring**: Health check endpoints for load balancers
- **Error Handling**: Comprehensive error responses and logging

### Web Interface Features

- **Interactive UI**: Gradio interface for non-technical users
- **Real-time Predictions**: Instant churn predictions with user-friendly output
- **Example Data**: Pre-filled examples for quick testing
- **Responsive Design**: Professional appearance with modern UI

### DevOps Features

- **Containerization**: Docker with optimized multi-stage builds
- **Cloud Deployment**: AWS ECS Fargate for serverless scaling
- **CI/CD Pipeline**: GitHub Actions for automated builds and deployments
- **Monitoring**: CloudWatch integration for logs and metrics
- **Security**: Proper security groups and network configuration

## 📁 Project Structure

```
End to End Machine learning Project/
├── 📁 src/                          # Source code
│   ├── 📁 app/                      # Web application
│   │   ├── app.py                   # FastAPI app configuration
│   │   └── main.py                  # Main application entry point
│   ├── 📁 data/                     # Data processing
│   │   ├── load_data.py             # Data loading utilities
│   │   └── preprocess.py            # Data preprocessing
│   ├── 📁 features/                 # Feature engineering
│   │   └── build_features.py        # Feature transformation pipeline
│   ├── 📁 models/                   # Model training and evaluation
│   │   ├── train.py                 # Model training
│   │   ├── evaluate.py              # Model evaluation
│   │   └── tune.py                  # Hyperparameter tuning
│   ├── 📁 serving/                  # Model serving
│   │   ├── inference.py             # Production inference pipeline
│   │   └── 📁 model/                # Trained model artifacts
│   └── 📁 utils/                    # Utility functions
│       ├── utils.py                 # General utilities
│       └── validate_data.py         # Data validation with Great Expectations
├── 📁 scripts/                      # Pipeline scripts
│   ├── run_pipeline.py              # Main training pipeline
│   ├── prepare_processed_data.py    # Data preparation
│   ├── test_fastapi.py              # API testing
│   ├── test_pipeline_phase1_data_features.py  # Phase 1 testing
│   └── test_pipeline_phase2_modeling.py       # Phase 2 testing
├── 📁 notebooks/                    # Jupyter notebooks
│   └── EDA.ipynb                    # Exploratory data analysis
├── 📄 requirements.txt              # Python dependencies
├── 📄 dockerfile                    # Docker configuration
└── 📄 README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (for containerization)
- AWS CLI (for deployment)
- Git

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/DarshanCode2005/Fullstack-ML-Churn-Prediction-Project.git
   cd "Fullstack-ML-Churn-Prediction-Project"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the training pipeline**
   ```bash
   python scripts/run_pipeline.py --input data/telco_churn.csv --experiment telco_churn
   ```

5. **Start the API server**
   ```bash
   python -m uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access the application**
   - API Documentation: http://localhost:8000/docs
   - Web UI: http://localhost:8000/ui
   - Health Check: http://localhost:8000/

### Docker Setup

1. **Build the Docker image**
   ```bash
   docker build -t telco-churn-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 telco-churn-api
   ```

3. **Test the API**
   ```bash
   python scripts/test_fastapi.py
   ```

## 📚 API Documentation

### Endpoints

#### Health Check
```http
GET /
```
Returns the health status of the API.

**Response:**
```json
{
  "status": "ok"
}
```

#### Predict Churn
```http
POST /predict
```
Predicts customer churn based on provided customer data.

**Request Body:**
```json
{
  "gender": "Male",
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "tenure": 5,
  "MonthlyCharges": 70.35,
  "TotalCharges": 350.75
}
```

**Response:**
```json
{
  "prediction": "Likely to churn"
}
```

#### Web Interface
```http
GET /ui
```
Access the Gradio web interface for interactive predictions.

### Data Schema

The API expects customer data with the following 18 features:

| Feature | Type | Description | Valid Values |
|---------|------|-------------|--------------|
| `gender` | string | Customer gender | "Male", "Female" |
| `Partner` | string | Has partner | "Yes", "No" |
| `Dependents` | string | Has dependents | "Yes", "No" |
| `PhoneService` | string | Has phone service | "Yes", "No" |
| `MultipleLines` | string | Has multiple lines | "Yes", "No", "No phone service" |
| `InternetService` | string | Internet service type | "DSL", "Fiber optic", "No" |
| `OnlineSecurity` | string | Online security service | "Yes", "No", "No internet service" |
| `OnlineBackup` | string | Online backup service | "Yes", "No", "No internet service" |
| `DeviceProtection` | string | Device protection service | "Yes", "No", "No internet service" |
| `TechSupport` | string | Tech support service | "Yes", "No", "No internet service" |
| `StreamingTV` | string | Streaming TV service | "Yes", "No", "No internet service" |
| `StreamingMovies` | string | Streaming movies service | "Yes", "No", "No internet service" |
| `Contract` | string | Contract type | "Month-to-month", "One year", "Two year" |
| `PaperlessBilling` | string | Paperless billing | "Yes", "No" |
| `PaymentMethod` | string | Payment method | "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)" |
| `tenure` | integer | Months with company | 0-120 |
| `MonthlyCharges` | float | Monthly charges ($) | 0-200 |
| `TotalCharges` | float | Total charges ($) | 0-10000 |

## 🤖 Model Training

### Training Pipeline

The training pipeline consists of several phases:

1. **Data Loading & Validation**
   ```bash
   python scripts/run_pipeline.py --input data/telco_churn.csv --experiment telco_churn
   ```

2. **Feature Engineering**
   - Binary encoding for 2-category features
   - One-hot encoding for multi-category features
   - Feature consistency validation

3. **Model Training**
   - XGBoost classifier with optimized hyperparameters
   - Cross-validation and performance evaluation
   - MLflow experiment logging

4. **Model Evaluation**
   - Classification metrics (precision, recall, F1-score, ROC-AUC)
   - Confusion matrix analysis
   - Feature importance analysis

### Hyperparameter Tuning

The project includes Optuna-based hyperparameter optimization:

```bash
python scripts/test_pipeline_phase2_modeling.py
```

### MLflow Integration

All experiments are tracked in MLflow with:
- Model parameters and hyperparameters
- Training and validation metrics
- Model artifacts and metadata
- Feature importance and evaluation results

## 🚀 Deployment

### AWS Deployment Architecture

```
Internet → ALB (Port 80) → ECS Fargate (Port 8000) → FastAPI App
                ↓
         CloudWatch Logs
```

### Deployment Steps

1. **Build and Push Docker Image**
   ```bash
   docker build -t your-username/telco-churn-api .
   docker push your-username/telco-churn-api
   ```

2. **Deploy to AWS ECS**
   - Create ECS cluster
   - Define task definition
   - Create service with ALB integration
   - Configure security groups

3. **Configure Load Balancer**
   - Application Load Balancer on port 80
   - Target group on port 8000
   - Health check on `/` endpoint

### CI/CD Pipeline

The project includes GitHub Actions workflow for:
- Automated Docker image builds
- Push to Docker Hub
- Optional ECS service updates

## 🛠️ Development

### Running Tests

```bash
# Test the complete pipeline
python scripts/test_pipeline_phase1_data_features.py
python scripts/test_pipeline_phase2_modeling.py

# Test the API
python scripts/test_fastapi.py
```

### Code Quality

The project follows best practices:
- Type hints throughout the codebase
- Comprehensive docstrings
- Error handling and logging
- Modular architecture
- Consistent naming conventions

### Adding New Features

1. **Data Processing**: Add new transformations in `src/data/` or `src/features/`
2. **Model Improvements**: Modify `src/models/` for new algorithms or techniques
3. **API Extensions**: Add new endpoints in `src/app/main.py`
4. **Validation**: Update `src/utils/validate_data.py` for new data checks

## 🔧 Troubleshooting

### Common Issues

#### 1. Module Import Errors
**Problem**: `ModuleNotFoundError: No module named 'src'`
**Solution**: Ensure `PYTHONPATH` includes the project root:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Model Loading Failures
**Problem**: Model not found during inference
**Solution**: Verify model artifacts are in the correct location:
```bash
ls -la src/serving/model/
```

#### 3. Data Validation Failures
**Problem**: Great Expectations validation errors
**Solution**: Check data quality and update validation rules in `src/utils/validate_data.py`

#### 4. Docker Build Issues
**Problem**: Container build failures
**Solution**: Check Dockerfile and ensure all dependencies are in `requirements.txt`

#### 5. AWS Deployment Issues
**Problem**: ECS tasks failing health checks
**Solution**: 
- Verify security group rules
- Check ALB target group configuration
- Ensure health check endpoint responds correctly

### Performance Optimization

- **Model Serving**: Use model quantization for faster inference
- **API Performance**: Implement caching for repeated predictions
- **Data Processing**: Use pandas optimizations and parallel processing
- **Container**: Multi-stage Docker builds for smaller images

### Monitoring and Logging

- **Application Logs**: CloudWatch integration for container logs
- **Metrics**: Custom metrics for prediction accuracy and latency
- **Alerts**: CloudWatch alarms for service health
- **Tracing**: Distributed tracing for request flow analysis

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at `/docs` endpoint

---

**Built with ❤️ for the ML community**
