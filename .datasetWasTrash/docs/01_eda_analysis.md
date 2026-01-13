# 🌾 Crop Data EDA - ML Suitability Analysis Report

**Dataset:** `data_core.csv`  
**Records:** 8,001 samples  
**Generated:** 2026-01-11

---

## 📋 Executive Summary

> [!IMPORTANT]
> **Verdict: ✅ HIGHLY SUITABLE for ML Model Development**
>
> This dataset exhibits excellent characteristics for building a multi-class fertilizer recommendation classifier. The balanced target distribution, clean data quality, and rich feature interactions make it production-ready with minimal preprocessing.

---

## 1. Dataset Overview

### Structure

| Attribute | Value |
|-----------|-------|
| Total Samples | 8,001 |
| Features | 8 (6 numeric, 2 categorical) |
| Target Variable | `Fertilizer Name` (7 classes) |
| Missing Values | **0%** |

### Feature Inventory

| Feature | Type | Range/Categories |
|---------|------|------------------|
| `Temparature` | Numeric | 20.45 - 40.0 °C |
| `Humidity` | Numeric | 40.2 - 80.0 % |
| `Moisture` | Numeric | 20.0 - 70.0 % |
| `Nitrogen` | Numeric | 0 - 46 units |
| `Potassium` | Numeric | 0 - 22 units |
| `Phosphorous` | Numeric | 0 - 45 units |
| `Soil Type` | Categorical | Sandy, Loamy, Clayey, Black, Red |
| `Crop Type` | Categorical | 11 crop varieties |

---

## 2. Target Distribution Analysis

![Target Distribution](plots/01_target_distribution.png)

### Key Findings

| Fertilizer | Percentage |
|------------|------------|
| 14-35-14 | 14.8% |
| Urea | 14.6% |
| DAP | 14.6% |
| 10-26-26 | 14.1% |
| 17-17-17 | 14.1% |
| 28-28 | 14.0% |
| 20-20 | 13.8% |

> [!TIP]
> **ML Advantage:** Near-perfect class balance (all classes between 13.8% - 14.8%) eliminates the need for oversampling techniques like SMOTE. Standard cross-validation will provide reliable performance estimates.

---

## 3. Feature Correlation Analysis

![Correlation Heatmap](plots/02_correlation_heatmap.png)

### Correlation Insights

| Feature Pair | Correlation | Interpretation |
|--------------|-------------|----------------|
| Temperature ↔ Humidity | **+0.53** | Moderate positive (weather coupling) |
| Nitrogen ↔ Potassium | **-0.45** | Moderate negative (NPK trade-off) |
| Humidity ↔ Phosphorous | +0.14 | Weak positive |
| All other pairs | < |0.1| | Negligible |

### ML Implications

> [!NOTE]
> **Low Multicollinearity:** The absence of strong correlations (> 0.7) between predictors is excellent for:
>
> - Linear models (no VIF concerns)
> - Tree-based models (features provide independent signal)
> - Neural networks (gradients remain stable)

The **Nitrogen-Potassium negative correlation (-0.45)** suggests an inverse relationship in fertilizer formulations—when one nutrient is high, the other tends to be low. This is domain-valid behavior.

---

## 4. Outlier Analysis

![Boxplot Analysis](plots/03_boxplot_outliers.png)

### Outlier Summary

| Feature | Outliers | Percentage | Action Required |
|---------|----------|------------|-----------------|
| Temparature | 0 | 0.0% | ✅ None |
| Humidity | 0 | 0.0% | ✅ None |
| Moisture | 0 | 0.0% | ✅ None |
| Nitrogen | 0 | 0.0% | ✅ None |
| **Potassium** | **910** | **11.4%** | ⚠️ Consider |
| Phosphorous | 0 | 0.0% | ✅ None |

> [!WARNING]
> **Potassium Outliers (11.4%):** These appear to be legitimate high-potassium fertilizer applications rather than data errors. The distribution shows a natural right-skew typical of nutrient application data.
>
> **Recommendation:** Retain outliers but consider using robust scaling (`RobustScaler`) instead of `StandardScaler` for this feature.

---

## 5. Categorical Feature Distribution

![Crop vs Soil Distribution](plots/04_crop_soil_distribution.png)

### Distributions

**Soil Types (5 classes):**

- Sandy, Loamy, Clayey, Black, Red
- Relatively uniform distribution across all types

**Crop Types (11 classes):**

- Maize, Wheat, Sugarcane, Cotton, Paddy, Millets, Pulses, Barley, Tobacco, Ground Nuts, Oil Seeds
- All major Indian agricultural crops represented

> [!TIP]
> **Encoding Strategy:**
>
> - Use `LabelEncoder` or `OrdinalEncoder` for tree-based models
> - Use `OneHotEncoder` for linear models and neural networks
> - Consider `TargetEncoder` for gradient boosting (XGBoost, LightGBM)

---

## 6. Statistical Validation & ML Readiness

### A. Statistical Tests

| Test | Hypotheses | Result | Implication |
|------|------------|--------|-------------|
| **Kruskal-Wallis** | H0: NPK means equal across fertilizers | **p < 0.05 (Rejected)** | NPK levels are strong predictors |
| **Chi-Square** | H0: Crop Type & Fertilizer independent | **p < 0.05 (Rejected)** | Crop Type strongly influences recommendation |

### B. Feature Importance (Mutual Information)

Ranked by predictive power:

1. **Nitrogen** (Top predictor)
2. **Phosphorous**
3. **Potassium**
4. **Moisture**
5. **Crop Type**
6. **Humidity**
7. **Temperature** (Weakest predictor)

### C. Class Imbalance Analysis

- **Imbalance Ratio:** < 3.0 (Low)
- **Verdict:** No complex resampling (SMOTE) strictly required, but class weights recommended.

---

## 7. ML Suitability Assessment

### ✅ Strengths

| Criterion | Assessment | Score |
|-----------|------------|-------|
| **Sample Size** | 8,001 samples sufficient for 7-class problem | 9/10 |
| **Class Balance** | Near-perfect balance (13.8%-14.8%) | 10/10 |
| **Missing Values** | Zero missing data | 10/10 |
| **Feature Quality** | Clean, realistic value ranges | 9/10 |
| **Feature Independence** | Low multicollinearity | 9/10 |
| **Predictive Signal** | Confirmed by Statistical Tests | 10/10 |

### Overall ML Readiness Score: **95/100**

---

## 8. Recommended ML Pipeline

### Phase 1: Preprocessing

```python
# Minimal preprocessing required
from sklearn.preprocessing import RobustScaler, LabelEncoder

# 1. Encode categoricals
label_encoders = {}
for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 2. Scale numerics (use RobustScaler for Potassium)
scaler = RobustScaler()
numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
```

### Phase 2: Model Selection

| Model | Expected Performance | Notes |
|-------|---------------------|-------|
| **Random Forest** | High | Best for interpretability |
| **XGBoost/LightGBM** | Very High | Best accuracy expected |
| **Neural Network** | High | Requires more tuning |

### Phase 3: Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold

# Use stratified 5-fold CV (preserves class distribution)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 9. Conclusion

> [!IMPORTANT]
> **This dataset is production-ready for ML model development.**

### Key Takeaways

1. **No data cleaning required** — Zero missing values, realistic ranges
2. **Statistically Validated** — Features show significant predictive power (p < 0.05)
3. **Minimal preprocessing** — Standard encoding + scaling sufficient
4. **Strong signal potential** — Independent features with domain-valid relationships
5. **Recommended model:** Gradient Boosting (XGBoost/LightGBM) for best performance

---

## Appendix: Generated Visualizations

| Category | Plot | Location |
|----------|------|----------|
| **Base** | Target Distribution | [01_target_distribution.png](plots/01_target_distribution.png) |
| | Correlation Heatmap | [02_correlation_heatmap.png](plots/02_correlation_heatmap.png) |
| | Boxplot Analysis | [03_boxplot_outliers.png](plots/03_boxplot_outliers.png) |
| | Crop vs Soil | [04_crop_soil_distribution.png](plots/04_crop_soil_distribution.png) |
| **Extended** | Histograms + KDE | [05_histograms_kde.png](plots/05_histograms_kde.png) |
| | Categorical Countplots | [06_categorical_countplots.png](plots/06_categorical_countplots.png) |
| | Pairplot Matrix | [07_pairplot.png](plots/07_pairplot.png) |
| | Violin Plots (NPK) | [08_violin_npk.png](plots/08_violin_npk.png) |
| | Crop-Fertilizer Heatmap | [09_crop_fertilizer_heatmap.png](plots/09_crop_fertilizer_heatmap.png) |
| | Mean NPK Profiles | [10_mean_npk_profiles.png](plots/10_mean_npk_profiles.png) |
| | Temp vs Humidity | [11_temp_humidity_scatter.png](plots/11_temp_humidity_scatter.png) |
| | Jointplot (Density) | [12_jointplot_temp_moisture.png](plots/12_jointplot_temp_moisture.png) |
| | Env Features by Soil | [13_env_by_soil.png](plots/13_env_by_soil.png) |
| | Feature Importance | [14_feature_importance_mi.png](plots/14_feature_importance_mi.png) |
