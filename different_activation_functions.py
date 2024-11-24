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
fig, axes = plt.subplots(len(activation_functions), 2, figsize=(12, 5 * len(activation_functions)))

# ---- Burası Aktivasyon Fonksiyonlara göre çıkan farklı sonuçları gösteriyor ----
for i, func in enumerate(activation_functions):
    print(f"\nTraining with {func} activation function:")

    # YSA
    nn = NeuralNetwork(input_size, hidden_size, output_size, func, lr=learning_rate, epochs=epochs)
    nn.train(X, Y)

    axes[i, 0].plot(range(epochs), nn.loss_history)
    axes[i, 0].set_title(f"Loss ({func})")
    axes[i, 0].set_xlabel("Epochs")
    axes[i, 0].set_ylabel("Loss")

    axes[i, 1].plot(range(epochs), nn.accuracy_history)
    axes[i, 1].set_title(f"Accuracy ({func})")
    axes[i, 1].set_xlabel("Epochs")
    axes[i, 1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()