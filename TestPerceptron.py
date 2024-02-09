# Example usage
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt
    import numpy as np
    from perceptron import Perceptron

    # Load a dataset (e.g., Iris)
    data = load_iris()
    X = data.data[:100, :2]  # Simplification: Use two features and two classes
    y = data.target[:100]
    y = np.where(y == 0, -1, 1)  # Convert to -1, 1 for the Perceptron

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Initialize and train the Perceptron
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)

    # Predict and evaluate
    predictions = p.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Optionally: Visualize the decision boundary and data points
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='b', label='1')
    plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='r', label='-1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend(loc='upper left')

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = p.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.show()
    