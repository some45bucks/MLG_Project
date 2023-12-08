import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool
from modified_sample_factory.algo.utils.torch_utils import calc_num_elements
from modified_sample_factory.model.encoder import Encoder, make_img_encoder
from modified_sample_factory.model.decoder import Decoder
from modified_sample_factory.model.core import ModelCore
from modified_sample_factory.model.actor_critic import ActorCriticGeo
from modified_sample_factory.model.model_utils import nonlinearity
from modified_sample_factory.algo.utils.context import global_model_factory
from modified_sample_factory.utils.typing import Config, ObsSpace, ActionSpace
from modified_sample_factory.utils.utils import log
from torch_geometric.nn import GAT
from modified_sample_factory.algo.utils.tensor_dict import TensorDict
from torch_geometric.data import HeteroData, Batch
from typing import Dict, Optional
from torch_geometric.nn import  global_max_pool
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
import time

class DoomEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        # reuse the default image encoder
        self.basic_encoder = make_img_encoder(cfg, obs_space["obs"])
        self.encoder_out_size = self.basic_encoder.get_out_size()

        self.measurements_head = None
        if "measurements" in list(obs_space.keys()):
            self.measurements_head = nn.Sequential(
                nn.Linear(obs_space["measurements"].shape[0], 128),
                nonlinearity(cfg),
                nn.Linear(128, 128),
                nonlinearity(cfg),
            )
            measurements_out_size = calc_num_elements(self.measurements_head, obs_space["measurements"].shape)
            self.encoder_out_size += measurements_out_size

        log.debug("Policy head output size: %r", self.get_out_size())

    def forward(self, obs_dict):
        x = self.basic_encoder(obs_dict["obs"])

        if self.measurements_head is not None:
            measurements = self.measurements_head(obs_dict["measurements"].float())
            x = torch.cat((x, measurements), dim=1)

        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size

class DoomActorCritic(ActorCriticGeo):

    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        metadata,
        cfg: Config,
    ):
        super().__init__(model_factory,obs_space, action_space,metadata, cfg)

        channels_dict = {}

        for obs_name in obs_space.spaces:
            if "geo_N_" in obs_name:
                channels_dict[obs_name.split("geo_N_")[1]] = obs_space.spaces[obs_name].shape[1]
                
        self.GAT = HeteroGNN(metadata=self.metadata,in_channels_dict=channels_dict, hidden_channels=64, out_channels=32,num_layers=3)
        self.MLP_GEO = SimpleMLP(32*len(self.metadata[0]),64,16)
        self.MLP_RL = SimpleMLP(self.core.core_output_size,64,16)

    def create_hetro_graph(self,obs):
        data_list = []

        for i in range(obs["obs"].shape[0]):
            hetero_data_single = HeteroData()

            for name in obs.keys():
                if "geo" in name and "_N_" in name:
                    real_name = name.split("_")[-1]
                    arr = obs[name][i]
                    mask = (arr != -1).all(dim=1)
                    last_num = torch.where(mask)[0].max().item() + 1 if mask.any() else 0
                    hetero_data_single[real_name].x = arr[:last_num]
                elif "geo" in name and "_E_" in name:
                    real_parts = name.split("_")[-1].split("-")
                    arr = obs[name][i].int().t()
                    if len(arr) == 2:
                        mask = (arr[0, :] != -1) & (arr[1, :] != -1)
                        arr = arr[:,mask]

                    hetero_data_single[real_parts[0], real_parts[1], real_parts[2]].edge_index = arr
            data_list.append(hetero_data_single)
        return Batch.from_data_list(data_list,follow_batch=self.metadata[0])

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True)
        result["new_rnn_states"] = new_rnn_states
        return result

    def forward_head_geo(self, batch_graph) -> Tensor:
        batch_dict = {node_type: batch_graph[node_type].batch for node_type in batch_graph.x_dict.keys()}
        out = self.GAT(batch_graph.x_dict,batch_graph.edge_index_dict,batch_dict)
        all_embeddings = []

        for node_type in out:
            all_embeddings.append(out[node_type])

        concatenated_embeddings = torch.cat(all_embeddings, dim=1)

        return concatenated_embeddings
        
    
    def forward_core_geo(self, head_output, head_geo_output):
        embedd_rl = self.MLP_RL(head_output)
        embedd_geo = self.MLP_GEO(head_geo_output)

        mat = torch.mm(embedd_rl,embedd_geo.t())

        mat = F.softmax(mat,dim=0)

        return mat



class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        for layer in range(num_layers):
            convs = {}
            for edge_type in metadata[1]:
                src_type, _, dst_type = edge_type
                src_in_channels = in_channels_dict[src_type] if layer == 0 else hidden_channels
                dst_in_channels = in_channels_dict[dst_type] if layer == 0 else hidden_channels

                convs[edge_type] = GATConv(
                    (src_in_channels, dst_in_channels), 
                    hidden_channels, 
                    add_self_loops=False
                )

            self.convs.append(HeteroConv(convs, aggr='sum'))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict,batch_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        pooled_x_dict = {key: global_max_pool(x, batch_dict[key]) for key, x in x_dict.items()}

        return {key: self.lin(x) for key, x in pooled_x_dict.items()}

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def make_doom_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace,metadata) -> DoomActorCritic:
    from modified_sample_factory.algo.utils.context import global_model_factory

    model_factory = global_model_factory()

    return DoomActorCritic(model_factory, obs_space, action_space,metadata, cfg)

def make_doom_encoder(cfg: Config, obs_space: ObsSpace) -> DoomEncoder:
    return DoomEncoder(cfg, obs_space)

def register_model_components():
    global_model_factory().register_encoder_factory(make_doom_encoder)
    global_model_factory().register_actor_critic_geo_factory(make_doom_actor_critic)
    