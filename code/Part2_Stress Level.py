# Import libraries
import pandas as pd
import numpy as np

# Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Library to handle imbalanced datasets
from imblearn.over_sampling import SMOTE

# sklearn libraries for data preprocessing, model selection, and model evaluation
from sklearn.preprocessing import LabelEncoder, train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

# sklearn classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Model selection and evaluation tools
from sklearn.model_selection import RandomizedSearchCV

from pathlib import Path

# Function to load data
def load_data():
    """
    Loads the health data from a CSV file.
    :return DataFrame(health)
    """

    DATA_ROOT = Path(__file__).parents[0] / "data"
    PATH_TO_HEALTH_DATA = (DATA_ROOT / "Sleep_health_and_lifestyle_dataset.csv").resolve()
    health = pd.read_csv(PATH_TO_HEALTH_DATA)
    return health

# Function to categorize mood based on stress level
def categorize_mood(stress_level):

    """
    Categorizes the mood based on the stress level.
    :param stress_level (int): The stress level value.
    :return str: Categorized mood as 'Calm', 'Neutral', or 'Overwhelmed'.
    """

    if stress_level <= 5:
        return 'Calm'
    elif 5 < stress_level < 7:
        return 'Neutral'
    else:
        return 'Overwhelmed'
    
# Function to preprocess data
def preprocess_data(health):
    """
    Preprocesses the health dataset for analysis and modeling.
    :param health (DataFrame): The health dataset.
    :return DataFrame: Preprocessed health data.
    """

    health['Stress Level'] = health['Stress Level'].apply(categorize_mood)
    health['BMI Category'].replace('Normal Weight', 'Normal', inplace=True)
    health[['Systolic BP', 'Diastolic BP']] = health['Blood Pressure'].str.split('/', expand=True)
    health['Systolic BP'] = pd.to_numeric(health['Systolic BP'])
    health['Diastolic BP'] = pd.to_numeric(health['Diastolic BP'])
    health.drop(['Person ID', 'Blood Pressure'], inplace=True, axis=1)
    health['Stress Level'] = health['Stress Level'].map({'Overwhelmed': 1, 'Neutral': 2, 'Calm': 3})
    categorical_cols = health.select_dtypes(include=['object']).columns
    for col in categorical_cols:      # looping through categorical columns
        health[col] = LabelEncoder().fit_transform(health[col])
    return health

# Function to train a Decision Tree model
def train_decision_tree(X_train, y_train):
    """
    Trains a Decision Tree Classifier.
    :param X_train: Training features.
    :param y_train: Training target variable.
    :return model: Trained Decision Tree model.
    """
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    return dt_model

# Function to train a Random Forest model
def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Classifier with hyperparameter tuning.
    :param X_train: Training features.
    :param y_train: Training target variable.
    :return model: Trained and tuned Random Forest model.
    """

    param_grid = {
        'n_estimators': [100, 200, 300],        # Number of trees in the forest
        'max_depth': [10, 20, 30, None],        # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],        # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],          # Minimum number of samples required to be at a leaf node
        'bootstrap': [True, False]              # Selecting samples for training each tree
    }
    # Initialize a RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    # Perform a randomized search over the parameter grid
    random_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Extract and print the best parameters found by RandomizedSearchCV
    best_params = random_search.best_params_
    print("Best Parameters: ", best_params)

    # Re-training the Tuned Model using best parameters

    rf_model = RandomForestClassifier(**best_params, random_state=42)
    rf_model.fit(X_train, y_train)

    return rf_model

# Function to train a Logistic Regression model
def train_logistic_regression(X_train, y_train, penalty):
    """
    Trains a Logistic Regression model with specified penalty(l1 or l2).
    :param X_train: Training features.
    :param y_train: Training target variable.
    :param penalty: The penalty type ('l1' or 'l2').
    :return model: Trained Logistic Regression model.
    """

    lr_model = LogisticRegression(penalty=penalty, multi_class='multinomial', random_state=42, max_iter=10000)
    lr_model.fit(X_train, y_train)
    return lr_model

# Function to train a Gradient Boosting model
def train_gradient_boosting(X_train, y_train):
    """
    Trains a Gradient Boosting Classifier.
    :param X_train: Training features.
    :param y_train: Training target variable.
    :return model: Trained and Tuned Gradient Boosting model.
    """

    # Initialize a Gradient Boosting Classifier with specified hyperparameters
    gbm_model = GradientBoostingClassifier(
        n_estimators=100,            # Number of boosting stages
        learning_rate=0.1,           # Lower learning rate
        max_depth=3,                 # Limit the depth of trees
        min_samples_leaf=4,          # Minimum samples per leaf
        random_state=42
    )
    # Train the model
    gbm_model.fit(X_train, y_train)
    return gbm_model

# Function to evaluate and print the model's performance
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints the classification report.
    :param model: The trained machine learning model.
    :param X_test: Test features.
    :param y_test: Test target variable.
    """

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Function for model comparison    
def compare_and_visualize_models(model_predictions, y_test, model_names):
    """
    Compares and visualizes the performance of different models.
    
    :param model_predictions: A dictionary containing model names as keys and their predictions as values.
    :param y_test: Actual test labels.
    :param model_names: List of model names.
    """
    # Store the evaluation metrics for each model
    model_metrics = {model: precision_recall_fscore_support(y_test, predictions, average='weighted') 
                     for model, predictions in model_predictions.items()}

    # Convert metrics to a DataFrame for easier visualization
    metrics_df = pd.DataFrame(model_metrics, index=['Precision', 'Recall', 'F1-Score', 'Support']).T.drop(columns='Support')

    # Plotting Precision, Recall, and F1-Score for each model using Matplotlib
    plt.figure(figsize=(10, 6))
    metrics_df[['Precision', 'Recall', 'F1-Score']].plot(kind='bar')
    plt.title('Model Comparison - Precision, Recall, F1-Score')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.show()

# Main function
def main():
    health = load_data()
    health = preprocess_data(health)

    X_train, X_test, y_train, y_test = train_test_split(health.drop('Stress Level', axis=1), health['Stress Level'], test_size=0.3, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_new, y_train_new = smote.fit_resample(X_train, y_train)

    # Train and evaluate Decision Tree
    dt_model = train_decision_tree(X_train_new, y_train_new)
    evaluate_model(dt_model, X_test, y_test)

    # Train and evaluate Random Forest
    rf_model = train_random_forest(X_train_new, y_train_new)
    evaluate_model(rf_model, X_test, y_test)

    # Train and evaluate Logistic Regression
    lr_model_l1 = train_logistic_regression(X_train_new, y_train_new, penalty='l1')
    evaluate_model(lr_model_l1, X_test, y_test)

    lr_model_l2 = train_logistic_regression(X_train_new, y_train_new, penalty='l2')
    evaluate_model(lr_model_l2, X_test, y_test)

    # Train and evaluate Gradient Boosting
    gb_model = train_gradient_boosting(X_train_new, y_train_new)
    evaluate_model(gb_model, X_test, y_test)

    # Prepare predictions for comparison
    model_predictions = {
        'Decision Tree': dt_model.predict(X_test),
        'Random Forest': rf_model.predict(X_test),
        'Logistic Regression L1': lr_model_l1.predict(X_test),
        'Logistic Regression L2': lr_model_l2.predict(X_test),
        'Gradient Boosting': gb_model.predict(X_test)
    }

    # Compare and visualize models
    model_names = ['Decision Tree', 'Random Forest', 'Logistic Regression L1', 'Logistic Regression L2', 'Gradient Boosting']
    compare_and_visualize_models(model_predictions, y_test, model_names)


if __name__ == "__main__":
    main()
