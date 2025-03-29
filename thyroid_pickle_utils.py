import pickle
import os
import pandas as pd
from sklearn.pipeline import Pipeline

def save_model(model, filename='thyroid_cancer_model.pkl'):
    """
    Save the trained model to disk using pickle
    
    Parameters:
    -----------
    model : object
        The trained model to save
    filename : str, default='thyroid_cancer_model.pkl'
        The path to save the model
    """
    print(f"\n--- Saving Model to {filename} ---")
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model successfully saved to {filename}")
    return

def load_model(filename='thyroid_cancer_model.pkl'):
    """
    Load a trained model from disk using pickle
    
    Parameters:
    -----------
    filename : str, default='thyroid_cancer_model.pkl'
        The path to the saved model
        
    Returns:
    --------
    model : object
        The loaded model
    """
    print(f"\n--- Loading Model from {filename} ---")
    if not os.path.exists(filename):
        print(f"Error: Model file {filename} not found")
        return None
        
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print("Model successfully loaded")
    return model

def thyroid_cancer_prediction(new_data, model):
    """
    Function to predict thyroid cancer risk on new data
    
    Parameters:
    -----------
    new_data : pandas.DataFrame or dict
        New patient data
    model : object
        Trained model
        
    Returns:
    --------
    results : pandas.DataFrame
        Predictions and probabilities
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

# Example usage
if __name__ == "__main__":
    # This code will only run if this script is executed directly
    # Path to the saved model
    model_path = 'thyroid_cancer_model.pkl'
    
    # Check if model exists
    if os.path.exists(model_path):
        # Load the model
        model = load_model(model_path)
        
        # Create sample patient data
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
        if model is not None:
            prediction = thyroid_cancer_prediction(sample_patient, model)
            print("\nPrediction result for sample patient:")
            print(prediction)
    else:
        print(f"Model file {model_path} not found. Run the thyroid_analysis.py script first to train and save the model.")