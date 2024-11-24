from functions import NeuralNetwork, InitWeights
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

#---- Burası Başlangıç Ağırlık değerlerine göre sigmoid fonksiyonunu kullanarak çıkan farklı sonuçları gösteriyor ----
init_methods = [InitWeights.init_weights_xavier, InitWeights.init_weights_he, 
                InitWeights.init_weights_random, InitWeights.init_weights_lecun, 
                InitWeights.init_weights_orthogonal]  
init_method_names = ['Xavier', 'He', 'Random', 'LeCun', 'Orthogonal']
fig, axes = plt.subplots(len(init_methods), 2, figsize=(12, 5 * len(init_methods)))

for i, (method, name) in enumerate(zip(init_methods, init_method_names)):
    print(f"\nTraining with {name} initialization:")
    nn = NeuralNetwork(input_size, hidden_size, output_size, 'sigmoid', lr=learning_rate, epochs=epochs, init_method=method)
    nn.train(X, Y)

    axes[i, 0].plot(range(epochs), nn.loss_history)
    axes[i, 0].set_title(f"Loss ({name})")
    axes[i, 0].set_xlabel("Epochs")
    axes[i, 0].set_ylabel("Loss")

    axes[i, 1].plot(range(epochs), nn.accuracy_history)
    axes[i, 1].set_title(f"Accuracy ({name})")
    axes[i, 1].set_xlabel("Epochs")
    axes[i, 1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()