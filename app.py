from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('svm_model.pkl')

FEATURES = [
    'radius_mean',
    'perimeter_mean',
    'area_mean',
    'compactness_mean',
    'concavity_mean',
    'concave_points_mean',
    'radius_se',
    'perimeter_se',
    'area_se',
    'radius_worst',
    'perimeter_worst',
    'area_worst',
    'compactness_worst',
    'concavity_worst',
    'concave_points_worst'
]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Extract features from form data
            input_data = [float(request.form[feature]) for feature in FEATURES]

            # Convert to numpy array and reshape
            data = np.array(input_data).reshape(1, -1)

            # Predict (assuming 0 = Malignant, 1 = Benign)
            pred = model.predict(data)[0]

            diagnosis = "Benign (No Cancer)" if pred == 1 else "Malignant (Cancer)"

            # Pass prediction but clear form inputs by not passing 'values'
            return render_template('index.html', features=FEATURES, prediction=diagnosis, values=None)

        except Exception as e:
            return f"Error: {str(e)}"
    else:
        return render_template('index.html', features=FEATURES, values=None)

if __name__ == '__main__':
    app.run(debug=True)

