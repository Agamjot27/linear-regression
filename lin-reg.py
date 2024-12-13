import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (2 / n_samples) * np.dot(X.T, (y - y_predicted))
            db = (2 / n_samples) * np.sum(y - y_predicted)
            self.weights += self.learning_rate * dw
            self.bias += self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def guess_salary(self, year):
        salary = year * self.weights + self.bias
        print(salary)
        print(f"Guessed salary for {year} years of experience --> {round(salary[0] / 1000, 2)}k")

if __name__ == "__main__":
    
    url = 'https://raw.githubusercontent.com/Agamjot27/<repolinear-regression/main/Salary_dataset.csv'

    
    data = pd.read_csv(url)

    X = data[['YearsExperience']].values  
    y = data['Salary'].values  

    model = LinearRegression(learning_rate=0.01, epochs=100)
    model.train(X, y)
    predictions = model.predict(X)
    model.guess_salary(4)

    print(f"Bias: {model.bias}")
    print(f"Weights: {model.weights}")
