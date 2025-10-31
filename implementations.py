import numpy as np

def compute_mse(y, tx, w):
    """Compute the mean squared error."""
    e = y - tx @ w
    return (1 / (2 * len(y))) * np.sum(e ** 2)

def compute_gradient(y, tx, w):
    """Compute the gradient for MSE."""
    e = y - tx @ w
    return - (1 / len(y)) * tx.T @ e

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (mini-batch size 1)."""
    w = initial_w
    n = len(y)
    for _ in range(max_iters):
        i = np.random.randint(0, n)
        grad = compute_gradient(y[i:i+1], tx[i:i+1], w)
        w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """Least squares regression using normal equations."""
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))

def compute_logistic_loss(y, tx, w):
    """Compute the negative log likelihood loss for logistic regression."""
    pred = sigmoid(tx @ w)
    epsilon = 1e-12
    loss = - (1 / len(y)) * (np.sum(y * np.log(pred + epsilon) + (1 - y) * np.log(1 - pred + epsilon)))
    return loss

def compute_logistic_gradient(y, tx, w, weights=None):
    """Compute the gradient for logistic regression, optionally with class weights."""
    pred = sigmoid(tx @ w)
    if weights is None:
        weights = np.ones(len(y))
    grad = (1 / len(y)) * tx.T @ ((pred - y) * weights)
    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_logistic_loss(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, weights=None):
    """Regularized logistic regression using gradient descent, optionally with class weights."""
    w = initial_w
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w, weights) + 2 * lambda_ * w
        w = w - gamma * grad
    loss = compute_logistic_loss(y, tx, w)
    return w, loss