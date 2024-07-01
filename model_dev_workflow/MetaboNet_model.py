import torch
import torch.nn as nn


class PriorKnowledgeLayer(nn.Module):
    def __init__(
        self,
        connectivity_matrix,
        use_bias=True,
        init_method="xavier_uniform",
        nonlinearity=None,  # Only used with kaiming
        device="cpu",
    ):
        super(PriorKnowledgeLayer, self).__init__()
        self.device = device
        self.connectivity_matrix = connectivity_matrix.to(device)  # Move to device
        self.input_size = self.connectivity_matrix.shape[0]
        self.output_size = self.connectivity_matrix.shape[1]
        self.use_bias = use_bias

        # Define weights only for the connections defined in the connectivity matrix
        self.weights = nn.Parameter(
            torch.zeros(self.input_size, self.output_size, device=device)
        )

        if init_method == "xavier_uniform":
            nn.init.xavier_uniform_(self.weights)
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(self.weights, nonlinearity=nonlinearity)

        self.weights.data *= self.connectivity_matrix

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.output_size, device=device))

    def forward(self, x):
        # Ensure only the defined connections are used
        connected_weights = self.weights * self.connectivity_matrix
        x = x @ connected_weights
        if self.use_bias:
            x = x + self.bias
        return x


class MetaboNet(nn.Module):
    def __init__(
        self,
        connectivity_matrices,
        l1_value,
        use_bias=True,
        hidden_layer_activation="Tanh",
        device="cpu",
    ):
        super(MetaboNet, self).__init__()

        self.l1_value = l1_value
        self.use_bias = use_bias
        self.device = device

        # Determine initialization method based on activation function
        if hidden_layer_activation == "Tanh" or hidden_layer_activation == "Sigmoid":
            init_method = "xavier_uniform"
            nonlinearity = None
        elif hidden_layer_activation == "ReLU":
            init_method = "kaiming_normal"
            nonlinearity = "relu"
        elif hidden_layer_activation == "PReLU":
            init_method = "kaiming_normal"
            nonlinearity = "leaky_relu"
        else:
            raise ValueError("Unsupported hidden layer activation function")

        # Layers
        self.hidden_layer1 = PriorKnowledgeLayer(
            connectivity_matrices["first_hidden_layer"],
            use_bias,
            init_method,
            nonlinearity,
            device=self.device,
        )
        self.hidden_layer2 = PriorKnowledgeLayer(
            connectivity_matrices["second_hidden_layer"],
            use_bias,
            init_method,
            nonlinearity,
            device=self.device,
        )
        self.hidden_to_output = nn.Linear(
            connectivity_matrices["second_hidden_layer"].shape[1], 1, bias=self.use_bias
        ).to(self.device)

        # Batch normalization
        self.batch_norm1 = nn.BatchNorm1d(
            connectivity_matrices["first_hidden_layer"].shape[0], affine=False
        ).to(self.device)
        self.batch_norm2 = nn.BatchNorm1d(
            connectivity_matrices["second_hidden_layer"].shape[0], affine=False
        ).to(self.device)
        self.batch_norm3 = nn.BatchNorm1d(
            connectivity_matrices["second_hidden_layer"].shape[1], affine=False
        ).to(self.device)

        # Activation functions
        self.hidden_layer_activation_str = hidden_layer_activation
        if hidden_layer_activation == "Tanh":
            self.hidden_layer_activation = nn.Tanh()
        elif hidden_layer_activation == "Sigmoid":
            self.hidden_layer_activation = nn.Sigmoid()
        elif hidden_layer_activation == "ReLU":
            self.hidden_layer_activation = nn.ReLU()
        elif hidden_layer_activation == "PReLU":
            self.hidden_layer_activation = nn.PReLU()

        self.final_layer_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.hidden_layer1(x)
        x = self.hidden_layer_activation(x)

        x = self.batch_norm2(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer_activation(x)

        x = self.batch_norm3(x)
        x = self.hidden_to_output(x)
        x = self.final_layer_activation(x)
        return x

    def l1_regularization(self):
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return self.l1_value * l1_norm


# if __name__ == "__main__":
#     # Example usage:
#     import numpy as np

#     first_layer_matrix = np.array(
#         [
#             [1, 0, 0],
#             [0, 1, 0],
#             [0, 1, 0],
#             [0, 0, 1],
#             [0, 1, 0],
#             [0, 0, 1],
#             [0, 0, 1],
#             [0, 0, 1],
#         ]
#     )
#     second_layer_matrix = np.array([[1, 0], [1, 0], [0, 1]])

#     connectivity_matrices = {
#         "first_hidden_layer": torch.tensor(first_layer_matrix, dtype=torch.float32),
#         "second_hidden_layer": torch.tensor(second_layer_matrix, dtype=torch.float32),
#     }

#     model_1 = BioArchitectureNetwork(connectivity_matrices, use_bias=True, l1_value=0.1)
#     model_2 = BioArchitectureNetwork(connectivity_matrices, use_bias=True, l1_value=0.1)
#     model_3 = BioArchitectureNetwork(connectivity_matrices, use_bias=True, l1_value=0.1)

#     model_states = [
#         model_1.state_dict().copy(),
#         model_2.state_dict().copy(),
#         model_3.state_dict().copy(),
#     ]

#     avg_model_state = average_model_states(model_states)

#     avg_model = BioArchitectureNetwork(
#         connectivity_matrices, use_bias=True, l1_value=0.1
#     )
#     print(model_1.state_dict())
#     print("=======")
#     print(model_2.state_dict())
#     print("=======")
#     print(model_3.state_dict())
#     print("=======")

#     avg_model.load_state_dict(avg_model_state)
#     print(avg_model.state_dict())

# # Input vector size 8, batch size 2
# input_data = torch.randn(2, 8)
# model = BioArchitectureNetwork(connectivity_matrices, use_bias=True, l1_value=0.1)
# output = model(input_data)
# print("output", output)

# # Print all model parameters
# for name, param in model.named_parameters():
#     print(f"Parameter name: {name}")
#     print(param.data)
