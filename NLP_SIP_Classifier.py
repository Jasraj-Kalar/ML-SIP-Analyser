"""
SIP Log Anomaly Detection & Classification

This script trains a model to classify SIP communication logs as normal or anomalous.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

CSV_FILENAME = 'Extracted_NLP_Features.csv'
plt.style.use('seaborn-v0_8-darkgrid')
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 12

def load_and_prepare_data(filepath):
    """Loads, cleans, and prepares the dataset for modeling."""
    print(f"Loading data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None, None, None

    # Clean and standardize data types to prevent errors
    for col in ['packets', 'time_diffs']:
        df[col] = df[col].fillna('').astype(str)
    for col in ['codec', 'final_response']:
        df[col] = df[col].fillna('N/A').astype(str)
    df['label'] = df['label'].astype(str)

    # Combine text feature for NLP processing
    df['combined_text'] = (df['packets'] + " " + df['time_diffs']).str.strip()
    
    # Define features (X) and target (y)
    X = df[['combined_text', 'codec', 'final_response']]
    y = df['label']
    
    # Encode the string labels into integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print("Data loading and preparation complete.\n")
    return X, y_encoded, label_encoder

def build_model_pipeline():
    """Defines the preprocessing steps and the model in a scikit-learn pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(ngram_range=(1, 2)), 'combined_text'),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), ['codec', 'final_response'])
        ],
        remainder='drop'
    )

    # Chain the preprocessor and the classifier together
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, C=1.0))
    ])
    return pipeline

def calculate_anomaly_scores(pipeline, X_test, y_test, y_pred):
    """
    Calculates an anomaly score based on the model's predicted confidence.
    Anomaly Score = 1 - (Probability of the predicted class)
    """
    # Get the probabilities for each class
    probabilities = pipeline.predict_proba(X_test)
    
    # Get the highest probability for each prediction
    confidence = np.max(probabilities, axis=1)
    
    anomaly_scores = 1 - confidence
    
    results_df = X_test.copy()
    results_df['anomaly_score'] = anomaly_scores
    results_df['is_correct'] = (y_pred == y_test)
    
    return results_df.sort_values(by='anomaly_score', ascending=False)
    
def plot_all_visualizations(pipeline, X_train, y_train, y_test, y_pred, anomaly_results, le):
    """Generates and displays the final 3 evaluation plots in a clean layout."""
    class_names = list(le.classes_)
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    fig.suptitle('Model Performance and Anomaly Analysis', fontsize=24, y=0.96)
    
    # Top-left plot
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_title('Confusion Matrix', fontsize=FONT_SIZE_TITLE)
    ax1.set_ylabel('Actual Label', fontsize=FONT_SIZE_LABEL)
    ax1.set_xlabel('Predicted Label', fontsize=FONT_SIZE_LABEL)

    # Top-right plot
    ax2 = fig.add_subplot(gs[0, 1])
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    ax2.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax2.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax2.set_title('Learning Curve', fontsize=FONT_SIZE_TITLE)
    ax2.set_xlabel('Training Examples', fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel('Accuracy Score', fontsize=FONT_SIZE_LABEL)
    ax2.legend(loc="best")
    ax2.grid(True)

    # Bottom plot
    ax3 = fig.add_subplot(gs[1, :])
    top_anomalies = anomaly_results.head(10)
    ax3.axis('off')
    ax3.set_title('Top 10 Most Anomalous Samples (Test Set)', fontsize=FONT_SIZE_TITLE, pad=20)
    table = ax3.table(
        cellText=top_anomalies[['anomaly_score', 'is_correct']].round(3).values,
        colLabels=['Anomaly Score', 'Was Correct?'],
        rowLabels=top_anomalies.index,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def main():
    """Main script."""
    # Load and prepare data
    X, y_encoded, label_encoder = load_and_prepare_data(CSV_FILENAME)
    if X is None:
        return
        
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    # Build and train the model
    pipeline = build_model_pipeline()
    print("Training the model...")
    pipeline.fit(X_train, y_train)
    print("Model training complete. ✅\n")
    
    # Evaluate the model on the test set
    print("Evaluating Model on Test Set")
    y_pred = pipeline.predict(X_test)
    class_names = list(label_encoder.classes_)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Calculate anomaly scores
    anomaly_results = calculate_anomaly_scores(pipeline, X_test, y_test, y_pred)
    
    # Generate all visualizations in one go
    print("Generating model performance board")
    plot_all_visualizations(pipeline, X_train, y_train, y_test, y_pred, anomaly_results, label_encoder)
    print("\nEnd")

if __name__ == '__main__':
    main()