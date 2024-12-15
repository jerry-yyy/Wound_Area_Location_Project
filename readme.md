# **Wound Area Location in Animal Model Images Project**

This is a university course project which implements machine learning models (Random Forest and XGBoost) to detect and localize wound areas in animal model images. The models predict the location and size of oval-shaped wound areas by determining four parameters: x-coordinate, y-coordinate, width, and height.

## Features

- Two ML models implemented: Random Forest and XGBoost
- Image preprocessing and data augmentation pipeline
- Model evaluation with MSE and overlap ratio metrics
- Visualization of predictions and error analysis
- Comprehensive analysis of difficult-to-predict cases

## Project Structure

```
├── Wound/
│   ├── Training/         # Training dataset
│   └── Test/            # Test dataset
├── train_rf.py          # Random Forest training script
├── train_xgboost.py     # XGBoost training script
├── test_rf.py           # Random Forest testing script
└── test_xgboost.py      # XGBoost testing script
```

## Getting Started

1. Clone the repository
```bash
git clone https://github.com/jerryyyy/Wound_area_location_project.git
cd Wound_area_location_project
```

3. Run training
```bash
python train_rf.py
python train_xgboost.py
```

4. Run testing
```bash
python test_rf.py
python test_xgboost.py
```
