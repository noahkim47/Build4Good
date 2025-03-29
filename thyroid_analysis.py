import pandas as pd
import pickle
import os

def save_model(model, filename='thyroid_cancer_model.pkl'):
    """
    Save the trained model to disk using pickle
    """
    print(f"\n--- Saving Model to {filename} ---")
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model successfully saved to {filename}")
    return

def load_model(filename='thyroid_cancer_model.pkl'):
    """
    Load a trained model from disk using pickle
    """
    print(f"\n--- Loading Model from {filename} ---")
    if not os.path.exists(filename):
        print(f"Error: Model file {filename} not found")
        return None
        
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print("Model successfully loaded")
    return model

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_explore_data(file_path):
    """
    Load and perform initial exploration of the dataset
    """
    # Load the data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Basic information
    print("\n--- Dataset Overview ---")
    print(f"Shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    
    # Data types and missing values
    print("\n--- Data Types and Missing Values ---")
    missing_data = df.isnull().sum()
    dtype_info = df.dtypes
    missing_percent = (missing_data / len(df)) * 100
    
    # Combine the information
    data_info = pd.DataFrame({
        'Data Type': dtype_info,
        'Missing Values': missing_data,
        'Missing Percent': missing_percent
    })
    print(data_info)
    
    # Summary statistics for numerical features
    print("\n--- Numerical Features Summary ---")
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    print(df[numerical_features].describe())
    
    # Distribution of categorical features
    print("\n--- Categorical Features Distribution ---")
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_features:
        print(f"\n{col} value counts:")
        print(df[col].value_counts())
        print(f"{col} value percentages:")
        print(df[col].value_counts(normalize=True) * 100)
    
    return df

def visualize_data(df):
    """
    Create interactive visualizations to understand the data better
    """
    print("\n--- Interactive Data Visualization ---")
    
    # Create interactive diagnostic plots
    
    # 1. Distribution of diagnosis (bar chart)
    diagnosis_counts = df['Diagnosis'].value_counts().reset_index()
    diagnosis_counts.columns = ['Diagnosis', 'Count']
    
    fig1 = px.bar(diagnosis_counts, x='Diagnosis', y='Count', 
                 title='Distribution of Diagnosis',
                 color='Diagnosis', template='plotly_white')
    fig1.write_html('interactive_diagnosis_distribution.html')
    print("Interactive diagnosis distribution saved as 'interactive_diagnosis_distribution.html'")
    
    # 2. Age distribution by diagnosis (box plot)
    fig2 = px.box(df, x='Diagnosis', y='Age', 
                 color='Diagnosis', title='Age Distribution by Diagnosis',
                 template='plotly_white')
    fig2.write_html('interactive_age_distribution.html')
    print("Interactive age distribution saved as 'interactive_age_distribution.html'")
    
    # 3. TSH Level by diagnosis (box plot)
    fig3 = px.box(df, x='Diagnosis', y='TSH_Level', 
                 color='Diagnosis', title='TSH Level by Diagnosis',
                 template='plotly_white')
    fig3.write_html('interactive_tsh_distribution.html')
    print("Interactive TSH distribution saved as 'interactive_tsh_distribution.html'")
    
    # 4. Nodule Size by diagnosis (box plot)
    fig4 = px.box(df, x='Diagnosis', y='Nodule_Size', 
                 color='Diagnosis', title='Nodule Size by Diagnosis',
                 template='plotly_white')
    fig4.write_html('interactive_nodule_size_distribution.html')
    print("Interactive nodule size distribution saved as 'interactive_nodule_size_distribution.html'")
    
    # 5. Interactive Correlation Matrix
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    
    fig5 = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                    title='Correlation Matrix of Numerical Features')
    fig5.write_html('interactive_correlation_matrix.html')
    print("Interactive correlation matrix saved as 'interactive_correlation_matrix.html'")
    
    # 6. Interactive Categorical Features vs Diagnosis
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'Diagnosis' in categorical_features:
        categorical_features.remove('Diagnosis')  # Remove target variable
    
    if categorical_features:
        for i, col in enumerate(categorical_features):
            # Create crosstab
            df_plot = pd.crosstab(df[col], df['Diagnosis'], normalize='index') * 100
            df_plot = df_plot.reset_index()
            df_plot_melted = pd.melt(df_plot, id_vars=col, var_name='Diagnosis', value_name='Percentage')
            
            # Create interactive bar chart
            fig = px.bar(df_plot_melted, x=col, y='Percentage', color='Diagnosis',
                        barmode='group', title=f'{col} vs Diagnosis',
                        template='plotly_white')
            
            fig.write_html(f'interactive_{col}_vs_diagnosis.html')
            print(f"Interactive {col} vs diagnosis plot saved as 'interactive_{col}_vs_diagnosis.html'")
    
    # 7. Create a dashboard of key variables (scatterplot matrix)
    key_vars = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
    fig7 = px.scatter_matrix(df, dimensions=key_vars, color='Diagnosis', 
                           opacity=0.7, title='Relationships Between Key Variables')
    fig7.write_html('interactive_scatterplot_matrix.html')
    print("Interactive scatterplot matrix saved as 'interactive_scatterplot_matrix.html'")
    
    return

def preprocess_data(df):
    """
    Preprocess the data for modeling
    """
    print("\n--- Data Preprocessing ---")
    
    # Drop Patient_ID as it's just an identifier
    if 'Patient_ID' in df.columns:
        df = df.drop('Patient_ID', axis=1)
    
    # Separate features and target
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    
    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    print(f"Numerical features: {list(numerical_features)}")
    print(f"Categorical features: {list(categorical_features)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Define preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

def build_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Build and evaluate multiple machine learning models
    """
    print("\n--- Model Building and Evaluation ---")
    
    # Dictionary to store model results
    model_results = {}
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if len(pipeline.classes_) == 2 else None
        
        # Calculate metrics
        print(f"\n{name} - Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\n{name} - Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Calculate and print ROC AUC if binary classification
        if len(np.unique(y_test)) == 2 and y_prob is not None:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            print(f"{name} - ROC AUC Score: {roc_auc:.4f}")
            
            # Store results
            model_results[name] = {
                'model': pipeline,
                'predictions': y_pred,
                'probabilities': y_prob,
                'roc_auc': roc_auc
            }
            
            # Plot static ROC curve
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=pipeline.classes_[1])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc='lower right')
            plt.savefig(f'roc_curve_{name.replace(" ", "_").lower()}.png')
            print(f"Static ROC curve saved as 'roc_curve_{name.replace(' ', '_').lower()}.png'")
            
            # Create interactive ROC curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{name} (AUC = {roc_auc:.4f})',
                line=dict(color='royalblue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Reference Line',
                line=dict(color='gray', width=2, dash='dash')
            ))
            fig.update_layout(
                title=f'ROC Curve - {name}',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                legend=dict(x=0.7, y=0.1),
                template='plotly_white'
            )
            fig.write_html(f'interactive_roc_curve_{name.replace(" ", "_").lower()}.html')
            print(f"Interactive ROC curve saved as 'interactive_roc_curve_{name.replace(' ', '_').lower()}.html'")
        else:
            # For multi-class
            model_results[name] = {
                'model': pipeline,
                'predictions': y_pred
            }
    
    return model_results

def feature_importance(model_results, X):
    """
    Extract feature importance from models and create interactive visualizations
    """
    print("\n--- Feature Importance Analysis ---")
    
    # Extract feature importance from Random Forest model if available
    if 'Random Forest' in model_results:
        # Get the trained pipeline
        pipeline = model_results['Random Forest']['model']
        
        # Get the random forest model from the pipeline
        rf_model = pipeline.named_steps['classifier']
        
        # Get feature names after preprocessing
        preprocessor = pipeline.named_steps['preprocessor']
        feature_names = []
        
        # Handle the case where preprocessor is a ColumnTransformer
        if hasattr(preprocessor, 'transformers_'):
            for name, transformer, features in preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(features)
                elif name == 'cat' and hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                    cat_features = transformer.named_steps['onehot'].get_feature_names_out(features)
                    feature_names.extend(cat_features)
        
        # If we can't get feature names, just use indices
        if not feature_names:
            feature_names = [f'Feature {i}' for i in range(rf_model.n_features_in_)]
        
        # Get feature importances
        importances = rf_model.feature_importances_
        
        # Create DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importances)],
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        # Display top 20 features
        print("\nTop 20 important features:")
        print(feature_importance_df.head(20))
        
        # Create interactive feature importance plot
        top_features = feature_importance_df.head(20).sort_values('Importance')
        fig = px.bar(top_features, x='Importance', y='Feature', 
                     title='Feature Importance (Random Forest)',
                     orientation='h', color='Importance',
                     color_continuous_scale='viridis',
                     template='plotly_white')
        
        fig.update_layout(
            height=800,
            xaxis_title='Importance',
            yaxis_title='Feature',
            coloraxis_showscale=False
        )
        
        fig.write_html('interactive_feature_importance.html')
        print("Interactive feature importance plot saved as 'interactive_feature_importance.html'")
        
        # Also save static version for compatibility
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
        plt.title('Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Static feature importance plot saved as 'feature_importance.png'")
    
    return

def tune_best_model(X_train, X_test, y_train, y_test, preprocessor, model_results):
    """
    Tune the best performing model using GridSearchCV
    """
    print("\n--- Model Tuning ---")
    
    # Find the best model based on ROC AUC (for binary classification)
    if all('roc_auc' in results for results in model_results.values()):
        best_model_name = max(model_results, key=lambda x: model_results[x]['roc_auc'])
    else:
        # If ROC AUC is not available, just use the first model
        best_model_name = list(model_results.keys())[0]
    
    print(f"Tuning the best model: {best_model_name}")
    
    # Define the parameter grid based on the best model
    if best_model_name == 'Logistic Regression':
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']
        }
    elif best_model_name == 'Random Forest':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_samples_split': [2, 5],
            'classifier__subsample': [0.8, 0.9, 1.0]
        }
    else:
        print(f"No parameter grid defined for {best_model_name}. Skipping tuning.")
        return model_results[best_model_name]['model']
    
    # Create a new pipeline with the same steps as the best model
    best_pipeline = model_results[best_model_name]['model']
    
    # Set up grid search
    grid_search = GridSearchCV(
        estimator=best_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc' if len(np.unique(y_train)) == 2 else 'accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    print(f"Running grid search with {len(param_grid)} parameter combinations...")
    grid_search.fit(X_train, y_train)
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    # Evaluate best model on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nFinal Model - Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nFinal Model - Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate and print ROC AUC if binary classification
    if len(np.unique(y_test)) == 2:
        y_prob = best_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        print(f"Final Model - ROC AUC Score: {roc_auc:.4f}")
        
        # Plot static ROC curve for final model
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=best_model.classes_[1])
        plt.plot(fpr, tpr, label=f'Final Model (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Final Model')
        plt.legend(loc='lower right')
        plt.savefig('roc_curve_final_model.png')
        print("Static final model ROC curve saved as 'roc_curve_final_model.png'")
        
        # Create interactive ROC curve for final model
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'Final Model (AUC = {roc_auc:.4f})',
            line=dict(color='royalblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Reference Line',
            line=dict(color='gray', width=2, dash='dash')
        ))
        fig.update_layout(
            title='ROC Curve - Final Tuned Model',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.7, y=0.1),
            template='plotly_white'
        )
        fig.write_html('interactive_roc_curve_final_model.html')
        print("Interactive final model ROC curve saved as 'interactive_roc_curve_final_model.html'")
    
    return best_model

def thyroid_cancer_prediction(new_data, model):
    """
    Function to predict thyroid cancer risk on new data
    """
    # Ensure new_data is a DataFrame
    if not isinstance(new_data, pd.DataFrame):
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
        else:
            raise ValueError("new_data must be a pandas DataFrame or a dictionary")
    
    # Make predictions
    predictions = model.predict(new_data)
    
    # Get probabilities if binary classification
    if hasattr(model, 'predict_proba') and model.predict_proba(new_data).shape[1] == 2:
        probabilities = model.predict_proba(new_data)[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Predicted_Diagnosis': predictions,
            'Probability': probabilities
        })
    else:
        results = pd.DataFrame({
            'Predicted_Diagnosis': predictions
        })
    
    return results

def create_interactive_dashboard(df, model_results, best_model):
    """
    Create an interactive dashboard HTML file combining key visualizations
    """
    print("\n--- Creating Interactive Dashboard ---")
    
    # Create a dashboard with multiple visualizations
    dashboard = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Diagnosis Distribution', 
            'Age Distribution by Diagnosis',
            'TSH Level by Diagnosis', 
            'Nodule Size by Diagnosis'
        ),
        specs=[
            [{"type": "bar"}, {"type": "box"}],
            [{"type": "box"}, {"type": "box"}]
        ]
    )
    
    # 1. Diagnosis counts
    diagnosis_counts = df['Diagnosis'].value_counts().reset_index()
    diagnosis_counts.columns = ['Diagnosis', 'Count']
    
    dashboard.add_trace(
        go.Bar(
            x=diagnosis_counts['Diagnosis'], 
            y=diagnosis_counts['Count'],
            marker_color=['#1f77b4', '#ff7f0e'],
            name='Diagnosis Distribution'
        ),
        row=1, col=1
    )
    
    # 2. Age distribution by diagnosis
    for diagnosis in df['Diagnosis'].unique():
        dashboard.add_trace(
            go.Box(
                y=df[df['Diagnosis'] == diagnosis]['Age'],
                name=diagnosis,
                boxmean=True
            ),
            row=1, col=2
        )
    
    # 3. TSH Level by diagnosis
    for diagnosis in df['Diagnosis'].unique():
        dashboard.add_trace(
            go.Box(
                y=df[df['Diagnosis'] == diagnosis]['TSH_Level'],
                name=diagnosis,
                boxmean=True
            ),
            row=2, col=1
        )
    
    # 4. Nodule Size by diagnosis
    for diagnosis in df['Diagnosis'].unique():
        dashboard.add_trace(
            go.Box(
                y=df[df['Diagnosis'] == diagnosis]['Nodule_Size'],
                name=diagnosis,
                boxmean=True
            ),
            row=2, col=2
        )
    
    # Update layout
    dashboard.update_layout(
        title_text="Thyroid Cancer Risk Analysis Dashboard",
        height=800,
        template='plotly_white',
        showlegend=False
    )
    
    # Save the dashboard
    dashboard.write_html('thyroid_analysis_dashboard.html')
    print("Interactive dashboard saved as 'thyroid_analysis_dashboard.html'")
    
    return

def main():
    """
    Main function to run the entire analysis pipeline
    """
    # Set file path
    file_path = 'thyroid_cancer_risk_data.csv'
    
    # Install required packages if not already installed
    try:
        import plotly
    except ImportError:
        print("Installing plotly package...")
        import pip
        pip.main(['install', 'plotly'])
        print("Plotly installed successfully.")
    
    # Load and explore data
    df = load_and_explore_data(file_path)
    
    # Visualize data
    visualize_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Build and evaluate models
    model_results = build_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
    
    # Analyze feature importance
    feature_importance(model_results, df)
    
    # Tune best model
    best_model = tune_best_model(X_train, X_test, y_train, y_test, preprocessor, model_results)
    
    # Save the best model to disk
    save_model(best_model, 'thyroid_cancer_model.pkl')
    
    # Create interactive dashboard
    create_interactive_dashboard(df, model_results, best_model)
    
    print("\n--- Analysis Complete ---")
    print("The best model has been saved and can be used for making predictions on new data.")
    print("\nTo use the saved model later:")
    print("1. Load the model: model = load_model('thyroid_cancer_model.pkl')")
    print("2. Create a new patient dataframe: new_patient = pd.DataFrame({...})")
    print("3. Make prediction: prediction = thyroid_cancer_prediction(new_patient, model)")
    print("\nInteractive visualizations have been saved as HTML files that you can open in any web browser.")
    
    return best_model

if __name__ == "__main__":
    # Run the full analysis pipeline and train models
    best_model = main()
    
    # Example of how to use the saved model
    print("\n--- Example: Loading and Using Saved Model ---")
    # Load the model
    loaded_model = load_model('thyroid_cancer_model.pkl')
    
    if loaded_model is not None:
        # Create a sample new patient data (replace with actual values)
        sample_patient = pd.DataFrame({
            'Age': [45],
            'Gender': ['Female'],
            'Country': ['USA'],
            'Ethnicity': ['Caucasian'],
            'Family_History': ['No'],
            'Radiation_Exposure': ['No'],
            'Iodine_Deficiency': ['No'],
            'Smoking': ['No'],
            'Obesity': ['No'],
            'Diabetes': ['No'],
            'TSH_Level': [2.5],
            'T3_Level': [110],
            'T4_Level': [7.8],
            'Nodule_Size': [1.2],
            'Thyroid_Cancer_Risk': ['Low']
        })
        
        # Make prediction
        prediction = thyroid_cancer_prediction(sample_patient, loaded_model)
        print("\nSample prediction result:")
        print(prediction)