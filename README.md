# 🌳 Deforestation Risk Classification Project
---

## 🌍 Why This Matters

Deforestation is one of the most pressing environmental challenges of our time. Every year, **10 million hectares** of forest—an area roughly the size of Iceland—are lost to logging, agriculture, and urban expansion. This destruction doesn't just harm trees; it:

- 💨 **Accelerates climate change** by releasing stored carbon
- 🦜 **Destroys biodiversity** and threatens countless species
- 🌊 **Disrupts water cycles** and increases flooding risks
- 👥 **Displaces indigenous communities** who depend on forests

**The problem?** By the time we detect deforestation through satellite imagery, the damage is already done.

**Our solution?** Use machine learning to **predict deforestation risk BEFORE it happens**, allowing conservationists to intervene proactively rather than reactively.

This project demonstrates how data science can transform environmental protection from a reactive practice into a predictive, preventative strategy.

---

## 📊 The Data

### Dataset Overview
- **Source**: Global country-level environmental and socioeconomic indicators, and annual deforestation (OurWorldInData)
- **Samples**: 138 observations (countries/regions)
- **Features**: 29 environmental, economic, and social variables
- **Target**: Binary classification (High Risk vs. Low Risk)
- **Class Distribution**: ~30% High Risk, ~70% Low Risk (imbalanced)

### Key Features Used
After feature engineering and selection, we focused on:
- 🌲 **Density (Population/km²)** - Human pressure on land
- 🌾 **Agricultural Land (%)** - Expansion pressure
- 🌍 **Absolute Latitude** - Distance from equator (climate proxy)
- 💰 **Economic indicators** - GDP, minimum wage, CPI
- 🎓 **Education levels** - Primary education enrollment
- 👶 **Social indicators** - Infant mortality, life expectancy
- ... and additional features

### Data Challenges
- **Small dataset** (n=138) - Risk of overfitting
- **Class imbalance** (70/30 split) - Required careful handling
- **Mixed feature scales** - Standardization
- **High dimensionality** - Feature selection was critical

---

## 🔬 Methodology

### 1. **Data Preprocessing**
```
✓ Train/Test Split (80/20) with stratification
✓ Feature engineering (Latitude → Abs_Latitude)
✓ Feature selection (dropped 3 low-importance features)
✓ Standardization (for distance-based models)
```

### 2. **Models Compared**
We implemented and rigorously tested **5 different algorithms**:

| Model | Type | Complexity | Best For |
|-------|------|------------|----------|
| **Logistic Regression** | Linear | Low | Baseline, interpretability |
| **Linear Discriminant Analysis (LDA)** | Linear | Low | Small datasets, assumes normality |
| **Random Forest** | Ensemble (Trees) | High | Non-linear patterns |
| **Support Vector Machine (SVM)** | Kernel-based | Medium | Small datasets, clear margins |
| **Neural Networks** | Deep Learning | High | Complex patterns (2 architectures tested) |

### 3. **Rigorous Evaluation Protocol**

To ensure fair comparison and prevent data leakage, we followed academic best practices:

**Phase 1: Model Comparison (Cross-Validation Only)**
- ✅ 5-fold stratified cross-validation on training data
- ✅ Hyperparameter tuning via GridSearchCV
- ✅ Class weight optimization (handling imbalance)
- ✅ Metrics: Recall, Precision, F1-Score, Accuracy
- ❌ **NO test set evaluation** (held out for final model only)

**Phase 2: Final Model Selection**
- Selected SVM based on **highest F1-score (0.723)** and **best recall-precision balance**
- Justification: Superior generalization for small datasets, mathematically stable decision boundary

**Phase 3: Final Evaluation**
- ✅ Test set evaluation (only AFTER model selection)
- ✅ Real-world performance estimation
- ✅ Interpretable results for stakeholders

### 4. **Special Considerations for Small Datasets**

Given our limited data (n=138), we took special precautions:

- **LDA**: Used shrinkage regularization (0.9) to stabilize covariance matrix
- **Random Forest**: Aggressive depth limiting (max_depth=5) to prevent overfitting
- **Neural Networks**: Minimalist architectures (16→8→1), early stopping, class weighting
- **SVM**: Linear kernel (avoids overfitting from polynomial/RBF kernels)

---

## 📈 Model Performance

### Cross-Validation Results (Training Data)

| Model | Recall | Precision | F1-Score | Accuracy |
|-------|--------|-----------|----------|----------|
| **SVM** ⭐ | **80.0%** | **68.5%** | **72.3%** | **81.9%** |
| **LDA** | **80.0%** | 65.4% | 70.8% | 79.1% |
| **Logistic Regression** | **80.0%** | 46.5% | 58.8% | 62.8% |
| **Random Forest** | 56.0% | 67.3% | 57.7% | 76.3% |
| **Neural Network (v1)** | 56.0% | 65.5% | 54.8% | 74.0% |
| **Neural Network (v2)** | 56.0% | 51.7% | 52.0% | 67.6% |

⭐ **Winner: Linear SVM** - Best overall F1-score and balanced performance (but in the end it is all contextual, we are just talking metrics for our main and target class aka the class 1)

### Key Insights

**✅ What Worked:**
- Linear models (LR, LDA, SVM) outperformed complex models (RF, NN)
- Class weighting successfully addressed the 70/30 imbalance
- Shrinkage regularization prevented overfitting in LDA
- Simple architectures worked best for limited data

**❌ What Didn't Work:**
- Deep learning struggled with small dataset (overfitting risk)
- Random Forest's complexity led to lower recall
- Aggressive class weights (balanced) caused precision collapse

**🔍 Domain Context:**
- **Recall prioritized**: Missing a high-risk area (false negative) = ecological disaster
- **Precision matters too**: False alarms waste conservation resources
- **F1-Score balance**: SVM's 72.3% represents optimal tradeoff

---

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.8+** - Programming language
- **NumPy & Pandas** - Data manipulation and analysis
- **Scikit-Learn** - Machine learning models and pipelines
- **TensorFlow/Keras** - Neural network implementations
- **Matplotlib & Seaborn** - Static visualizations

### Interactive Components
- **Streamlit** - Web app framework for Model Arena
- **Plotly** - Interactive visualizations and dashboards

### Development Tools
- **Jupyter Notebook** - Analysis and experimentation
- **Git** - Version control
- **Google Colab** - Cloud computing (initial development)

### Key Techniques
- Stratified K-Fold Cross-Validation
- Grid Search Hyperparameter Tuning
- Pipeline Architecture (prevent data leakage)
- Class Weight Balancing
- Regularization (L2, shrinkage)
- Early Stopping & Learning Rate Scheduling

---

## 🎮 Interactive App: Model Arena

Want to **play with the models** and **predict deforestation risk** yourself?

We built a **fun, interactive Streamlit app** with two modes:

### 🥊 Mode 1: Model Arena
**"Pick your fighters and watch them battle!"**

- Select any 2 models (e.g., SVM vs. Random Forest)
- Click "BATTLE!" 
- See head-to-head metric comparison
- Declares a winner based on F1-score
- Visual battle statistics chart

Perfect for understanding model tradeoffs!

### 🔮 Mode 2: Risk Predictor
**"Predict deforestation risk for any region!"**

- Adjust sliders for all environmental/economic features
- Get instant HIGH RISK 🚨 or LOW RISK ✅ prediction
- See confidence scores
- Get actionable recommendations
- Compare predictions across all models

Perfect for testing hypothetical scenarios!

### How to Launch

```bash
streamlit run model_arena_app.py
```

Or just double-click:
```bash
./launch_arena.sh
```

Opens in browser at `localhost`

---

## Project Files:
* [DeforestationClassifier.ipynb](./DeforestationClassifier.ipynb): The full analysis, feature engineering, model testing, metric analysis, etc.
* [DeforestationClassifier.html](./DeforestationClassifier.html): The ready-to-share version of the results.
* [classified_deforestation.csv](./classified_deforestation.csv): The dataset.
* [model_arena_app.py](./model_arena_app.py): the interactive streamlit website with the predictive tool and the fighting ring.
* [launch_arena.sh](./launch_arena.sh): The simple way to launch (if on your computer just clicik play or double click on the file)

---

<div align="center">

**Built with 💚 for the Planet**

</div>
