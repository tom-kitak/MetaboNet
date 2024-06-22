import pandas as pd
import numpy as np
import torch
from bio_model_v11 import PriorKnowledgeLayer, BioArchitectureNetwork


def calculate_bionet_importance(model):

    importance = dict()
    # Metabolite
    importance["metabolite"] = (
        model.hidden_layer1.weights @ model.hidden_layer2.weights
    ) @ model.hidden_to_output.weight.T

    importance["metabolite"] = torch.abs(importance["metabolite"]) / torch.sum(
        torch.abs(importance["metabolite"])
    )

    # Sub-pathway
    connectivity_matrix_l1 = model.hidden_layer1.connectivity_matrix
    importance["sub_pathway"] = torch.zeros(connectivity_matrix_l1.shape[1])

    for col in range(connectivity_matrix_l1.shape[1]):
        importance["sub_pathway"][col] = (
            connectivity_matrix_l1[:, col] @ importance["metabolite"]
        )

    importance["sub_pathway"] = torch.abs(importance["sub_pathway"]) / torch.sum(
        torch.abs(importance["sub_pathway"])
    )

    # Super-pathway
    connectivity_matrix_l2 = model.hidden_layer2.connectivity_matrix
    importance["super_pathway"] = torch.zeros(connectivity_matrix_l2.shape[1])

    for col in range(connectivity_matrix_l2.shape[1]):
        importance["super_pathway"][col] = (
            connectivity_matrix_l2[:, col] @ importance["sub_pathway"]
        )

    importance["super_pathway"] = torch.abs(importance["super_pathway"]) / torch.sum(
        torch.abs(importance["super_pathway"])
    )

    # Normalize
    # importance["metabolite_percentage"] = torch.abs(
    #     importance["metabolite"]
    # ) / torch.sum(torch.abs(importance["metabolite"]))
    # importance["sub_pathway_percentage"] = torch.abs(
    #     importance["sub_pathway"]
    # ) / torch.sum(torch.abs(importance["sub_pathway"]))
    # importance["super_pathway_percentage"] = torch.abs(
    #     importance["super_pathway"]
    # ) / torch.sum(torch.abs(importance["super_pathway"]))

    return importance


#  Testing code:
if __name__ == "__main__":
    seed = 3
    np.random.seed(seed)
    torch.manual_seed(seed)

    first_layer_matrix = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )
    second_layer_matrix = np.array([[1, 1], [1, 0], [0, 1]])

    connectivity_matrices = {
        "first_hidden_layer": torch.tensor(first_layer_matrix, dtype=torch.float32),
        "second_hidden_layer": torch.tensor(second_layer_matrix, dtype=torch.float32),
    }

    model = BioArchitectureNetwork(connectivity_matrices, use_bias=True, l1_value=0.1)

    print(calculate_bionet_importance(model))
