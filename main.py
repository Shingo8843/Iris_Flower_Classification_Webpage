from flask import Flask, request, jsonify, render_template
from perceptron import Perceptron  
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/run_model', methods=['POST'])
def run_model():
    data = request.json
    learning_rate = data['learningRate']
    iterations = data['iterations']
    
    iris = load_iris()
    X = iris.data[:100, :2]  
    y = iris.target[:100]
    y = np.where(y == 0, -1, 1)  

    # randomize split


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    perceptron = Perceptron(learning_rate=learning_rate, n_iters=iterations)
    perceptron.fit(X_train_scaled, y_train)

    predictions = perceptron.predict(X_test_scaled)
    accuracy = np.mean(predictions == y_test)

    x_values = np.linspace(X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1, 300)
    y_values = -(perceptron.weights[0] * x_values + perceptron.bias) / perceptron.weights[1]
    boundary_points = np.column_stack((x_values, y_values))

    return jsonify({
        'accuracy': accuracy * 100,
        'predictions': predictions.tolist(),
        'y_test': y_test.tolist(),
        'X_test': X_test_scaled.tolist(),
        'weights': perceptron.weights.tolist(),
        'bias': perceptron.bias,
        'boundaryPoints': boundary_points.tolist()  
    })

if __name__ == '__main__':
    app.run(debug=True)
