import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch.torch_utils import get_activation


class HRM_Block(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_units=[512, 256, 128],
                 hidden_activations="ReLU",
                 dropout_rate=0.0):
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = get_activation(hidden_activations, hidden_units)
        super(HRM_Block, self).__init__()
        self.layers = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for i in range(len(hidden_units) - 1):
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            if hidden_activations[i]:
                self.layers.append(hidden_activations[i])


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class HRM(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="SimCEN",
                 gpu=-1,
                 layers=[],
                 fc={},
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dropout_rate=0.1,
                 batch_norm=False,
                 layer_norm=False,
                 **kwargs):
        super(HRM, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  **kwargs)

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.input_dim = feature_map.num_fields * embedding_dim
        self.hrm_high_layers = nn.ModuleList()
        self.hrm_low_layers = nn.ModuleList()
        self.hrm_norms = nn.ModuleList()
        high_input_dim = 0
        low_input_dim = self.input_dim
        for layer in layers:
            self.hrm_high_layers.append(
                HRM_Block(input_dim=high_input_dim + self.input_dim, dropout_rate=dropout_rate, **layer["high"]))
            self.hrm_low_layers.append(HRM_Block(input_dim=low_input_dim, dropout_rate=dropout_rate, **layer["low"]))
            input_dim = layer["high"]["hidden_units"][-1] + layer["low"]["hidden_units"][-1]
            if batch_norm:
                self.hrm_norms.append(
                    nn.BatchNorm1d(input_dim))

            elif layer_norm:
                self.hrm_norms.append(
                    nn.LayerNorm(input_dim))
            high_input_dim = layer["high"]["hidden_units"][-1]
            low_input_dim = input_dim

        self.fc = nn.Linear(in_features=low_input_dim, out_features=1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, x):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(x)
        feature_emb = self.embedding_layer(X)
        feature_emb = feature_emb.flatten(start_dim=1)

        high_rep = feature_emb
        low_rep = feature_emb
        for i in range(len(self.hrm_high_layers)):
            high_rep = self.hrm_high_layers[i](high_rep)
            low_rep = self.hrm_low_layers[i](low_rep)
            combined_rep = torch.cat([high_rep, low_rep], dim=-1)
            if len(self.hrm_norms) > i:
                combined_rep = self.hrm_norms[i](combined_rep)
            high_rep = torch.cat([high_rep, feature_emb], dim=-1)
            low_rep = combined_rep

        y_pred = self.fc(low_rep)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict


class MLP_Block(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 output_dim=None,
                 output_activation=None,
                 dropout_rates=0.0,
                 batch_norm=False,
                 layer_norm=False,
                 norm_before_activation=True,
                 use_bias=True):
        super(MLP_Block, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = get_activation(hidden_activations, hidden_units)
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if norm_before_activation:
                if batch_norm:
                    dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
                elif layer_norm:
                    dense_layers.append(nn.LayerNorm(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if not norm_before_activation:
                if batch_norm:
                    dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
                elif layer_norm:
                    dense_layers.append(nn.LayerNorm(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.mlp = nn.Sequential(*dense_layers)  # * used to unpack list

    def forward(self, inputs):
        return self.mlp(inputs)
