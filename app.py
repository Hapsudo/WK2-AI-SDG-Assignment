from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

features = [
    'radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'radius_se', 'perimeter_se', 'area_se', 'radius_worst',
    'perimeter_worst', 'area_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Check for missing features
    if not all(f in data for f in features):
        missing = [f for f in features if f not in data]
        return jsonify({'error': f'Missing features: {missing}'}), 400
    
    # Extract features in correct order
    X = np.array([data[f] for f in features]).reshape(1, -1)
    
    # Scale input
    X_scaled = scaler.transform(X)
    
    # Predict class and probability
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][prediction]
    
    diagnosis_map = {0: "Malignant (Cancer)", 1: "Benign (No Cancer)"}
    
    return jsonify({
        'prediction': int(prediction),
        'diagnosis': diagnosis_map[prediction],
        'confidence': float(proba)
    })

if __name__ == '__main__':
    app.run(debug=True)
