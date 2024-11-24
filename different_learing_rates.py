from functions import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

X = np.load("X.npy")
Y = np.load("Y.npy")
X = X.reshape(X.shape[0], -1)

input_size = X.shape[1]
hidden_size = 128
output_size = Y.shape[1]
learning_rate = 0.1
epochs = 100

activation_functions = ['sigmoid', 'tanh', 'relu', 'leaky_relu']

learning_rates = [0.01, 0.1, 0.3, 0.7, 1.0]
learning_rate_names = ['0.01', '0.1', '0.3', '0.7', '1.0']
fig, axes = plt.subplots(len(learning_rates), 2, figsize=(12, 5 * len(learning_rates)))

#---- Farklı öğrenme katsayıları ile Relu fonksiyonunda eğitim sonuçlarını gösteriyor ----
for i, (lr, lr_name) in enumerate(zip(learning_rates, learning_rate_names)):
    print(f"Training with learning rate {lr_name}")
    nn = NeuralNetwork(input_size, hidden_size, output_size, 'relu', lr=lr, epochs=epochs)
    nn.train(X, Y)

    axes[i, 0].plot(range(0, len(nn.loss_history)*10, 10), nn.loss_history)
    axes[i, 0].set_title(f"Loss (LR={lr_name})")
    axes[i, 0].set_xlabel("Epochs")
    axes[i, 0].set_ylabel("Loss")

    axes[i, 1].plot(range(0, len(nn.accuracy_history)*10, 10), nn.accuracy_history)
    axes[i, 1].set_title(f"Accuracy (LR={lr_name})")
    axes[i, 1].set_xlabel("Epochs")
    axes[i, 1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()
