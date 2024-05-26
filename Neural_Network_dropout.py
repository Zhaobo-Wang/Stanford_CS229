import numpy as np

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input_data, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input_data.shape)
            return input_data * self.mask / (1 - self.dropout_rate)
        else:
            return input_data

    def backward(self, d_output):
        return d_output * self.mask / (1 - self.dropout_rate)

# Example neural network layer
class Dense:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def forward(self, input_data):
        self.input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, d_output):
        d_input = np.dot(d_output, self.weights.T)
        d_weights = np.dot(self.input.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= self.learning_rate * d_weights
        self.biases -= self.learning_rate * d_biases

        return d_input

# Example neural network with Dropout
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, learning_rate=0.01):
        self.hidden_layer = Dense(input_size, hidden_size, learning_rate)
        self.dropout = Dropout(dropout_rate)
        self.output_layer = Dense(hidden_size, output_size, learning_rate)

    def forward(self, input_data, training=True):
        hidden_output = self.hidden_layer.forward(input_data)
        dropout_output = self.dropout.forward(hidden_output, training)
        output = self.output_layer.forward(dropout_output)
        return output

    def backward(self, d_output):
        d_hidden = self.output_layer.backward(d_output)
        d_hidden = self.dropout.backward(d_hidden)
        self.hidden_layer.backward(d_hidden)

# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Hyperparameters
    input_size = 3
    hidden_size = 4
    output_size = 2
    dropout_rate = 0.5
    learning_rate = 0.01
    epochs = 10

    # Dummy data
    X = np.random.randn(5, input_size)
    y = np.random.randn(5, output_size)

    # Initialize the neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size, dropout_rate, learning_rate)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        predictions = nn.forward(X, training=True)
        
        # Compute the loss (mean squared error)
        loss = np.mean((predictions - y) ** 2)
        
        # Backward pass
        d_loss = 2 * (predictions - y) / y.size
        nn.backward(d_loss)
        
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    # Forward pass without dropout for evaluation
    predictions = nn.forward(X, training=False)
    print("Predictions (without dropout):", predictions)
