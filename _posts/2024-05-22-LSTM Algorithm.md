# LSTM Algorithm

LSTM (Long Short-Term Memory) networks are a type of recurrent neural network (RNN) capable of learning long-term dependencies. They were introduced by Hochreiter and Schmidhuber in 1997 and have since become very popular for various sequence modeling tasks. LSTMs are designed to overcome the limitations of traditional RNNs, particularly the issue of vanishing and exploding gradients, which makes it difficult for RNNs to learn long-range dependencies.

#### Core Concepts

1. **Memory Cell**:
   - The core component of an LSTM is the memory cell, which maintains its state over time and allows the network to remember or forget information.

2. **Gating Mechanisms**:
   - **Forget Gate**: Decides what information from the cell state should be discarded.
   - **Input Gate**: Determines what new information should be added to the cell state.
   - **Output Gate**: Controls what information from the cell state should be output.

These gates are implemented using sigmoid and tanh functions, which help regulate the flow of information through the network.

#### LSTM Equations

Given an input sequence \( x_t \), the LSTM computes the hidden state \( h_t \) and cell state \( c_t \) as follows:

1. **Forget Gate**:
   \[
   f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
   \]
   
2. **Input Gate**:
   \[
   i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
   \]
   \[
   \tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
   \]
   
3. **Cell State Update**:
   \[
   c_t = f_t \cdot c_{t-1} + i_t \cdot \tilde{c}_t
   \]
   
4. **Output Gate**:
   \[
   o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
   \]
   \[
   h_t = o_t \cdot \tanh(c_t)
   \]

Here, \( \sigma \) represents the sigmoid function, and \( \tanh \) represents the hyperbolic tangent function. \( W \) and \( b \) are the weights and biases of the gates.

#### Example

Below is a simple example of an LSTM implemented using PyTorch for a sequence prediction task.

```python
import torch
import torch.nn as nn

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define model parameters
input_dim = 10  # Dimension of input features
hidden_dim = 20  # Dimension of hidden state
num_layers = 2  # Number of LSTM layers
output_dim = 1  # Dimension of output

# Create the LSTM model
model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

# Define input data
batch_size = 5
sequence_length = 7
input_data = torch.rand(batch_size, sequence_length, input_dim)

# Forward pass
output = model(input_data)

print(output.shape)  # Output dimension: (batch size, output_dim)
```

### Explanation

1. **Model Definition**:
   - The `LSTMModel` class defines an LSTM with specified input dimensions, hidden state dimensions, number of layers, and output dimensions.
   - The LSTM layer is followed by a fully connected layer to produce the final output.

2. **Initialization**:
   - `h0` and `c0` are the initial hidden and cell states, initialized to zeros.
   - The input sequence is passed through the LSTM layer and then through the fully connected layer to produce the output.

3. **Forward Pass**:
   - The input data, shaped as (batch size, sequence length, input_dim), is fed into the model.
   - The output is the prediction for each sequence in the batch.

This basic LSTM example can be extended for various tasks such as time series forecasting, language modeling, and more by adjusting the input and output dimensions and adding additional layers or regularization techniques as needed.
