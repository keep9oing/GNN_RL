import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_add_pool

import numpy as np
import random
from itertools import permutations

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# REINFROCE Network
class REINFORCE_graph(nn.Module):
    def __init__(self, state_space=None,
                       action_space=None,
                       num_hidden_layer=2,
                       hidden_dim=None,
                       learning_rate=None):

        super(REINFORCE_graph, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        assert action_space is not None, "None action_space input: action_space should be assigned"

        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.conv1 = GCNConv(2, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, action_space)
        self.layer_norm = LayerNorm(hidden_dim)

        self.roll_out = []
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def put_data(self, data):
        self.roll_out.append(data)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = global_add_pool(self.layer_norm(x), torch.LongTensor([0 for _ in range(4)]).to(device))
        x = F.relu(self.linear(x))
        x = self.linear2(x)
        out = F.log_softmax(x, dim=1)
        return out

    def train_net(self, gamma):
        R = 0
        G = []
        G_t = 0
        
        # Whitening baseline
        for r, prob in self.roll_out[::-1]:
            G_t = r + gamma * G_t
            G.append(G_t)
        G = np.array(G)
        G_mean = G.mean()
        G_std  = G.std()
        self.optimizer.zero_grad()

        for r, prob in self.roll_out[::-1]:
            R = r + gamma * R
            loss = -prob * ((R-G_mean) / G_std)
            loss.backward()
        self.optimizer.step()
        self.roll_out = []


def create_torch_graph_data(data):

    edge_index = list(permutations([i for i in range(4)], 2))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()

    node_feature = [[data[0], data[1]],[data[2], data[3]],[data[0], data[3]],[data[1], data[2]]]
    # node_feature = [[data[0], data[3]],[data[1], data[2]]]
    node_feature = torch.tensor(node_feature, dtype=torch.float)

    data = Data(x=node_feature, edge_index=edge_index)

    return data
    
def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)

if __name__ == "__main__":

    # Determine seeds
    model_name = "REINFORCE"
    env_name = "CartPole-v1"
    seed = 10
    exp_num = 'SEED_'+str(seed)

    # Set gym environment
    env = gym.make(env_name)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    # set parameters
    learning_rate = 0.0005
    episodes = 100000
    discount_rate = 0.99
    print_interval = 10

    Policy = REINFORCE_graph(state_space=env.observation_space.shape[0],
                             action_space=env.action_space.n,
                             num_hidden_layer=0,
                             hidden_dim=128,
                             learning_rate=learning_rate).to(device)

    score = 0

    for epi in range(episodes):
        s = env.reset()
        done = False

        step = 0

        while not done:
            # if epi%print_interval == 0:
            #     env.render()

            # Get action
            s_g = create_torch_graph_data(s)

            a_prob = Policy(s_g.x.to(device), s_g.edge_index.to(device))
            a_distrib = Categorical(torch.exp(a_prob))
            a = a_distrib.sample()

            # Interaction with Environment
            s_prime, r, done, _ = env.step(a.item())

            Policy.put_data((r, a_prob[0][a]))
            s = s_prime
            score += r
            step += 1
        
        Policy.train_net(discount_rate)

        # Logging/
        if epi%print_interval==0 and epi!=0:
            print("# of episode :{}, avg score : {}".format(epi, score/print_interval))
            score = 0.0

    env.close()