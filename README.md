# AgroSense: Intelligent Crop Recommendation Engine

AgroSense is a diagnostic platform for agricultural precision, leveraging a multi-model ensemble to recommend optimal crops based on soil and environmental metrics (N, P, K, pH, rainfall, temperature, and humidity). The system integrates Explainable AI (xAI) to provide transparency into predictive modeling.

---

## 🛠 Core Technical Implementation

### 🧠 Ensemble Prediction Architecture

- **Consensus Matrix**: Aggregates predictions from five distinct model architectures (Random Forest, XGBoost, SVM, CatBoost, and Ensemble Voting) to provide a high-confidence suitability score.
- **Cross-Model Validation**: Performs automated verification between models to ensure prediction consistency and reduce outlier risk.

### 🧪 Explainable AI & Scenario Analysis

- **xAI Integration**: Native support for SHAP-based feature importance, visualizing the quantitative impact of each soil metric on a per-prediction basis.
- **What-If Multi-Simulator**: An interactive scenaraio modeling interface allowing agronomists to simulate the impact of shifted soil health parameters.

### 📊 Historical Intelligence Dashboard

- **Dynamic Soil Analytics**: Automatically aggregates soil health trends (e.g., pH deviation, Nitrogen deficiency) from historical analysis data.
- **Contextual Insights**: Condition-specific agricultural guidance generated from the aggregated analysis history.

### 🎨 Unified Design System

- **Compact UI Standard**: High-density diagnostic interface optimized for professional workflows.
- **Visual Integrity**: Standardized 8px border-radius (`rounded-lg`) and glassmorphic styling enforced across all dashboard and analysis modules.

---

## 💻 Technology Stack

- **Frontend**: Next.js 15 (App Router), TypeScript, Framer Motion, TanStack Query v5, Lucide Icons.
- **Backend API**: FastAPI (Python 3.10+), Pydantic V2 validation, Async IO.
- **Machine Learning**: Scikit-learn, XGBoost, Joblib.
- **Infrastructure**: Multi-container Docker orchestration.

---

## 🚀 Deployment & Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)
- Node.js 18+ & Python 3.10+ (for local development)

### Quick Start (Containerized)

The recommended way to deploy the AgroSense ecosystem is via the orchestrated Docker stack:

```bash
# Clone the repository
git clone https://github.com/haiderCho/AgroSense.git
cd AgroSense

# Build and launch services
docker-compose up --build
```

- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **API (Swagger Docs)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 Project Governance

```text
AgroSense/
├── backend/             # FastAPI High-Throughput Inference Engine
│   ├── api/             # RESTful Routes & Pydantic Schemas
│   ├── inference/       # Ensemble Pipeline & xAI Logic
│   └── models/          # Model Factory & Configurations
├── frontend/            # Next.js Diagnostic Interface
│   ├── src/features/    # Domain-Driven Modular Components
│   └── src/app/         # Routing Architecture & Page Definitions
├── models/              # Persistent ML Model Weights
│   ├── catboost/        # CatBoost Model Artifacts
│   ├── ensemble/        # Voting Ensemble Weights
│   └── ...              # Other Model Weights (RF, XGB, SVM)
├── notebooks/           # Data Science & EDA Notebooks
├── scripts/             # Training & Utility Scripts
└── docker-compose.yml   # Multi-service Orchestration
```

---
*AgroSense — Engineered for sustainable precision agriculture.*
