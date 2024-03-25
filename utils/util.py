import os
import numpy as np
import torch

def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def save_networks(model, communication_idx):
    nets_list = model.nets_list
    model_name = model.NAME

    checkpoint_path = model.checkpoint_path
    model_path = os.path.join(checkpoint_path, model_name)
    model_para_path = os.path.join(model_path, 'para')
    create_if_not_exists(model_para_path)
    for net_idx, network in enumerate(nets_list):
        each_network_path = os.path.join(model_para_path, str(communication_idx) + '_' + str(net_idx) + '.ckpt')
        torch.save(network.state_dict(), each_network_path)


        
def generate_online_clients_sequence(epochs, parti_num, online_ratio):
    sequence = {}
    for epoch in range(epochs):
        total_clients = list(range(parti_num))
        online_clients = np.random.choice(total_clients, int(parti_num * online_ratio), replace=False).tolist()
        sequence[epoch] = online_clients
    return sequence
