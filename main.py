import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Function to load datasets
def get_dataset(name):
    data = datasets.load_iris() if name == 'Iris' else datasets.load_wine() if name == 'Wine' else datasets.load_breast_cancer()
    return data.data, data.target, data

# Model building function
def build_model(classifier_name, params):
    if classifier_name == 'SVM':
        model = SVC(C=params['C'], kernel=params['kernel'], probability=True)
    elif classifier_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=123)
    else:  # Gradient Boosting
        model = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], max_depth=params['max_depth'], random_state=123)
    return model

# Function to plot ROC Curve
def plot_roc_curve(fpr, tpr):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

# Function to plot feature importances for tree-based models
def plot_feature_importances(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        st.pyplot(plt)

# Function to plot data distributions
def plot_data_distributions(X, y, feature_names):
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.pairplot(df, hue='target', diag_kind='kde')
    st.pyplot()

# UI layout
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .emoji {
        font-size:20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="big-font">Machine Learning Model Explorer 🚀</div>', unsafe_allow_html=True)
st.markdown('<div class="big-font">Explore different machine learning models and see how they perform on various datasets 🛳️</div>', unsafe_allow_html=True)

# Sidebar for dataset selection
with st.sidebar:
    st.markdown('## Build your model 🏗️')
    dataset_name = st.selectbox('Select Dataset', ('Iris', 'Wine', 'Breast Cancer'))
    classifier_name = st.selectbox('Select Classifier', ('SVM', 'Random Forest', 'Gradient Boosting'))

    # Classifier parameters
    params = {}
    if classifier_name == 'SVM':
        params['C'] = st.number_input('C (Regularization parameter)', 0.01, 10.0, step=0.01, value=1.0)
        params['kernel'] = st.radio('Kernel', ('rbf', 'linear'))
    elif classifier_name == 'Random Forest':
        params['n_estimators'] = st.slider('Number of trees in the forest', 10, 100, step=10, value=10)
        params['max_depth'] = st.slider('Maximum depth of the tree', 1, 20, step=1, value=5)
    else:  # Gradient Boosting
        params['n_estimators'] = st.slider('Number of boosting stages', 10, 100, step=10, value=10)
        params['learning_rate'] = st.number_input('Learning Rate', 0.01, 1.0, step=0.01, value=0.1)
        params['max_depth'] = st.number_input('Maximum depth of the tree', 1, 20, step=1, value=5)

# Load and split the dataset
X, y, dataset = get_dataset(dataset_name)
st.write('Dataset Shape:', X.shape)
st.write('Number of classes:', len(np.unique(y)))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Visualize dataset distributions
if st.button('Show Data Distributions'):
    plot_data_distributions(X, y, dataset.feature_names)

# Build and train the model
model = build_model(classifier_name, params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Performance metrics
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Display metrics with emojis
st.markdown('### Model Performance Metrics:')
st.markdown(f'* Accuracy: {acc:.2f} ✅')
st.markdown(f'* Precision: {precision:.2f} 🎯')
st.markdown(f'* Recall: {recall:.2f} 🔍')
st.markdown(f'* F1 Score: {f1:.2f} ⚖️')

# ROC Curve (for binary classification)
if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plot_roc_curve(fpr, tpr)

# Feature importances for tree-based models
if classifier_name in ['Random Forest', 'Gradient Boosting']:
    if st.button('Show Feature Importances'):
        plot_feature_importances(model, dataset.feature_names)

# Model descriptions
with st.expander("Learn More About The Models"):
    if classifier_name == 'SVM':
        st.write("Support Vector Machine (SVM) details...")
    elif classifier_name == 'Random Forest':
        st.write("Random Forest details...")
    else:  # Gradient Boosting
        st.write("Gradient Boosting details...")

# Run this from the terminal to start the Streamlit app
# streamlit run this_script.py
