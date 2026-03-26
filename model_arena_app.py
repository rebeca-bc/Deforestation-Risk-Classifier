"""
🌳 DEFORESTATION MODEL ARENA 🥊
A fun, interactive app to compare models and predict deforestation risk!
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🌳 Model Arena",
    page_icon="🥊",
    layout="wide"
)

# ============================================================================
# LOAD DATA & TRAIN MODELS (cached so it only runs once)
# ============================================================================

@st.cache_resource
def load_models():
    """Load data and train all models - only runs once!"""
    
    # Load data
    df = pd.read_csv('classified_deforestation.csv')
    X = df.drop('Deforestation_Critical', axis=1)
    y = df['Deforestation_Critical']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Drop columns (same as notebook)
    columns_to_drop = ['Total tax rate', 'Gross tertiary education enrollment (%)', 'Co2-Emissions']
    X_train_final = X_train.drop(columns=columns_to_drop)
    X_test_final = X_test.drop(columns=columns_to_drop)
    
    # Transform latitude
    X_train_final['Latitude'] = X_train_final['Latitude'].abs()
    X_test_final['Latitude'] = X_test_final['Latitude'].abs()
    X_train_final = X_train_final.rename(columns={'Latitude': 'Abs_Latitude'})
    X_test_final = X_test_final.rename(columns={'Latitude': 'Abs_Latitude'})
    
    # Get feature names
    selected_features = X_train_final.columns.tolist()
    
    # Train models
    models = {}
    
    # Logistic Regression
    models['Logistic Regression'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=0.5, random_state=42, max_iter=1000))
    ])
    models['Logistic Regression'].fit(X_train_final, y_train)
    
    # LDA
    models['LDA'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearDiscriminantAnalysis(shrinkage=0.9, solver='lsqr'))
    ])
    models['LDA'].fit(X_train_final, y_train)
    
    # Random Forest
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_split=10,
        min_samples_leaf=5, random_state=42, class_weight='balanced'
    )
    models['Random Forest'].fit(X_train_final, y_train)
    
    # SVM
    models['SVM'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(C=10, kernel='linear', class_weight='balanced', random_state=42, probability=True))
    ])
    models['SVM'].fit(X_train_final, y_train)
    
    return models, X_test_final, y_test, selected_features

# Load everything
models, X_test, y_test, feature_names = load_models()

# Model stats (from your CV results)
model_stats = {
    'Logistic Regression': {'Recall': 0.800, 'Precision': 0.465, 'F1': 0.588, 'Accuracy': 0.628},
    'LDA': {'Recall': 0.800, 'Precision': 0.654, 'F1': 0.708, 'Accuracy': 0.791},
    'Random Forest': {'Recall': 0.560, 'Precision': 0.673, 'F1': 0.577, 'Accuracy': 0.763},
    'SVM': {'Recall': 0.800, 'Precision': 0.685, 'F1': 0.723, 'Accuracy': 0.819}
}

# ============================================================================
# HEADER
# ============================================================================

st.title("🌳 Deforestation Model Arena 🥊")
st.markdown("### *Pick your fighters and watch them battle!*")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2 = st.tabs(["🥊 Model Arena", "🔮 Risk Predictor"])

# ============================================================================
# TAB 1: MODEL ARENA
# ============================================================================

with tab1:
    st.header("⚔️ Choose Your Fighters!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔴 Fighter 1")
        model1 = st.selectbox(
            "Select Model 1",
            ['Logistic Regression', 'LDA', 'Random Forest', 'SVM'],
            key='model1'
        )
    
    with col2:
        st.subheader("🔵 Fighter 2")
        model2 = st.selectbox(
            "Select Model 2",
            ['Logistic Regression', 'LDA', 'Random Forest', 'SVM'],
            key='model2',
            index=3  # Default to SVM
        )
    
    if st.button("⚔️ BATTLE!", use_container_width=True, type="primary"):
        st.markdown("---")
        st.subheader("🎮 BATTLE RESULTS!")
        
        # Get stats
        stats1 = model_stats[model1]
        stats2 = model_stats[model2]
        
        # Display head-to-head
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown(f"### 🔴 {model1}")
            st.metric("🎯 Recall", f"{stats1['Recall']:.1%}")
            st.metric("🔍 Precision", f"{stats1['Precision']:.1%}")
            st.metric("⚖️ F1-Score", f"{stats1['F1']:.1%}")
            st.metric("📊 Accuracy", f"{stats1['Accuracy']:.1%}")
        
        with col2:
            st.markdown("### VS")
            st.markdown("")
            st.markdown("### ⚔️")
            
        with col3:
            st.markdown(f"### 🔵 {model2}")
            st.metric("🎯 Recall", f"{stats2['Recall']:.1%}")
            st.metric("🔍 Precision", f"{stats2['Precision']:.1%}")
            st.metric("⚖️ F1-Score", f"{stats2['F1']:.1%}")
            st.metric("📊 Accuracy", f"{stats2['Accuracy']:.1%}")
        
        # Determine winner
        st.markdown("---")
        
        # Compare F1 scores
        if stats1['F1'] > stats2['F1']:
            winner = model1
            emoji = "🔴"
            margin = (stats1['F1'] - stats2['F1']) * 100
        elif stats2['F1'] > stats1['F1']:
            winner = model2
            emoji = "🔵"
            margin = (stats2['F1'] - stats1['F1']) * 100
        else:
            winner = "TIE"
            emoji = "🤝"
            margin = 0
        
        if winner != "TIE":
            st.success(f"# {emoji} **{winner} WINS!** {emoji}")
            st.info(f"Victory margin: {margin:.1f} F1-Score points!")
        else:
            st.info(f"# {emoji} **IT'S A TIE!** {emoji}")
        
        # Visual comparison
        st.markdown("---")
        st.subheader("📊 Battle Statistics")
        
        metrics = ['Recall', 'Precision', 'F1', 'Accuracy']
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=model1,
            x=metrics,
            y=[stats1[m] for m in metrics],
            marker_color='#FF6B6B'
        ))
        
        fig.add_trace(go.Bar(
            name=model2,
            x=metrics,
            y=[stats2[m] for m in metrics],
            marker_color='#4ECDC4'
        ))
        
        fig.update_layout(
            barmode='group',
            yaxis_title='Score',
            yaxis_range=[0, 1],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: RISK PREDICTOR
# ============================================================================

with tab2:
    st.header("🔮 Predict Deforestation Risk")
    st.markdown("*Adjust the sliders to see predictions from all models!*")
    
    # Model selector
    prediction_model = st.selectbox(
        "Choose prediction model:",
        ['SVM', 'LDA', 'Random Forest', 'Logistic Regression'],
        help="SVM is recommended (best F1-score)"
    )
    
    st.markdown("---")
    
    # Create input sliders for all features
    st.info(f"📊 Adjust values for {len(feature_names)} features")
    
    # Create sliders dynamically for all features
    input_values = {}
    
    # Split into 2 columns
    col1, col2 = st.columns(2)
    
    for idx, feature in enumerate(feature_names):
        col = col1 if idx % 2 == 0 else col2
        with col:
            # Set reasonable defaults based on feature name
            if 'density' in feature.lower() or 'population' in feature.lower():
                val = st.slider(f"📊 {feature}", 0.0, 500.0, 100.0, 5.0, key=feature)
            elif 'latitude' in feature.lower():
                val = st.slider(f"🌍 {feature}", 0.0, 90.0, 30.0, 1.0, key=feature)
            elif '%' in feature or 'percent' in feature.lower():
                val = st.slider(f"📈 {feature}", 0.0, 100.0, 50.0, 1.0, key=feature)
            else:
                val = st.slider(f"⚙️ {feature}", 0.0, 100.0, 50.0, 1.0, key=feature)
            input_values[feature] = val
    
    # Create input dataframe with actual features
    input_data = pd.DataFrame([input_values])
    
    # Make prediction
    if st.button("🔮 PREDICT RISK!", use_container_width=True, type="primary"):
        st.markdown("---")
        
        # Get prediction
        prediction = models[prediction_model].predict(input_data)[0]
        
        # Get probability if available
        if hasattr(models[prediction_model], 'predict_proba'):
            proba = models[prediction_model].predict_proba(input_data)[0]
            confidence = max(proba) * 100
        else:
            # For models without predict_proba, use decision_function or just show prediction
            confidence = 85  # Placeholder
        
        # Display result
        if prediction == 1:
            st.error(f"# 🚨 HIGH RISK - {confidence:.0f}% confidence")
            st.warning("⚠️ This area shows concerning deforestation indicators!")
            st.markdown("""
            **Recommended Actions:**
            - 🔍 Immediate field inspection required
            - 📊 Enhanced satellite monitoring
            - 🛡️ Consider protective measures
            """)
        else:
            st.success(f"# ✅ LOW RISK - {confidence:.0f}% confidence")
            st.info("👍 This area appears stable with minimal deforestation threat.")
            st.markdown("""
            **Recommendations:**
            - 📅 Regular monitoring schedule
            - 🌱 Continue conservation efforts
            - 📈 Track trends over time
            """)
        
        # Show all model predictions
        st.markdown("---")
        st.subheader("🤖 All Model Predictions")
        
        cols = st.columns(4)
        for idx, (name, model) in enumerate(models.items()):
            pred = model.predict(input_data)[0]
            with cols[idx]:
                if pred == 1:
                    st.error(f"**{name}**\n\n🚨 High Risk")
                else:
                    st.success(f"**{name}**\n\n✅ Low Risk")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Made with 🌳 for Deforestation Classification Project | 
    Data: classified_deforestation.csv
</div>
""", unsafe_allow_html=True)
