from sklearn.datasets import load_boston
from model import PolynomialBostonPredictor

if __name__ == "__main__":
    # Load the Boston Housing Price dataset
    boston = load_boston()

    # Separate features and target variable
    X = boston.data
    y = boston.target

    # Instantiate the model
    model = PolynomialBostonPredictor(degree=2)

    # Train the model
    model.train(X, y)

    # Evaluate the model
    evaluation_results = model.evaluate(X, y)
    
    for metric, value in evaluation_results.items():
        print(f'{metric}: {value:.3f}')
