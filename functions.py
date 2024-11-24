import numpy as np

class InitWeights:
    @staticmethod
    def init_weights_xavier(m):
        if isinstance(m, np.ndarray): 
            return np.random.randn(*m.shape) * np.sqrt(1 / m.shape[0]) 
        return m

    @staticmethod
    def init_weights_he(m):
        if isinstance(m, np.ndarray):  
            return np.random.randn(*m.shape) * np.sqrt(2 / m.shape[0]) 
        return m

    @staticmethod
    def init_weights_random(m):
        if isinstance(m, np.ndarray):
            return np.random.uniform(-0.1, 0.1, size=m.shape) 
        return m

    @staticmethod
    def init_weights_lecun(m):
        if isinstance(m, np.ndarray): 
            return np.random.normal(0, np.sqrt(1 / m.shape[0]), size=m.shape)  
        return m

    @staticmethod
    def init_weights_orthogonal(m):
        if isinstance(m, np.ndarray): 
            rows, cols = m.shape
            a = np.random.randn(rows, rows) 
            q, r = np.linalg.qr(a) 
            if q.shape == m.shape:
                return q
            return q[:rows, :cols]
        return m

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_function, lr=0.01, epochs=100, init_method=None):

        self.params = {
            "W1": np.random.randn(input_size, hidden_size) * 0.01,
            "b1": np.zeros((1, hidden_size)),
            "W2": np.random.randn(hidden_size, output_size) * 0.01,
            "b2": np.zeros((1, output_size))
        }

        self.cache = {}
        self.grads = {}
        self.activation_function = activation_function
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []
        self.accuracy_history = []

        #farklı başlangıç ağırlıkları (init_method boş gelmiyorsa)
        if init_method:
            self.params["W1"] = init_method(self.params["W1"])
            self.params["W2"] = init_method(self.params["W2"])

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z)**2

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def leaky_relu(self, z):
        return np.maximum(0.01 * z, z)

    def leaky_relu_derivative(self, z):
        return (z > 0).astype(float) + 0.01 * (z <= 0).astype(float)

    def activation(self, Z):
        if self.activation_function == 'sigmoid':
            return self.sigmoid(Z)
        elif self.activation_function == 'tanh':
            return self.tanh(Z)
        elif self.activation_function == 'relu':
            return self.relu(Z)
        elif self.activation_function == 'leaky_relu':
            return self.leaky_relu(Z)

    def activation_derivative(self, Z):
        if self.activation_function == 'sigmoid':
            return self.sigmoid_derivative(Z)
        elif self.activation_function == 'tanh':
            return self.tanh_derivative(Z)
        elif self.activation_function == 'relu':
            return self.relu_derivative(Z)
        elif self.activation_function == 'leaky_relu':
            return self.leaky_relu_derivative(Z)

    def forward(self, X):
        Z1 = np.dot(X, self.params["W1"]) + self.params["b1"]
        A1 = self.activation(Z1)
        Z2 = np.dot(A1, self.params["W2"]) + self.params["b2"]
        A2 = self.softmax(Z2) 

        self.cache = {"A1": A1, "A2": A2, "Z1": Z1, "Z2": Z2}
        return A2

    def backward(self, X, Y):
        m = X.shape[0]
        A1, A2 = self.cache["A1"], self.cache["A2"]

        dZ2 = A2 - Y
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = np.dot(dZ2, self.params["W2"].T) * self.activation_derivative(self.cache["Z1"])
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_parameters(self):
        self.params["W1"] -= self.lr * self.grads["dW1"]
        self.params["b1"] -= self.lr * self.grads["db1"]
        self.params["W2"] -= self.lr * self.grads["dW2"]
        self.params["b2"] -= self.lr * self.grads["db2"]

    def compute_loss(self, Y, A2):
        m = Y.shape[0]
        loss = -np.mean(np.sum(Y * np.log(A2), axis=1)) 
        return loss
    
    def compute_accuracy(self, Y, A2):
        predictions = np.argmax(A2, axis=1)
        true_labels = np.argmax(Y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

    def train(self, X, Y):
        for epoch in range(self.epochs):
            A2 = self.forward(X)
            loss = self.compute_loss(Y, A2)
            self.backward(X, Y)
            self.update_parameters()

            self.loss_history.append(loss)
            accuracy = self.compute_accuracy(Y, A2)
            self.accuracy_history.append(accuracy)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}")
