import numpy as np
import torch.nn as nn
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from utils.conf import checkpoint_path
from utils.util import create_if_not_exists
import os
import copy

class FederatedModel(nn.Module):
    """
    Federated learning model.
    """
    NAME = None
    N_CLASS = None

    def __init__(self, nets_list: list,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(FederatedModel, self).__init__()
        self.nets_list = nets_list
        self.args = args
        self.transform = transform

        self.random_state = np.random.RandomState()
        self.online_num = np.ceil(self.args.parti_num * self.args.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.device = get_device(device_id=self.args.device_id)
        self.freq = None
        self.local_epoch = args.local_epoch
        self.local_lr = args.local_lr
        self.online_clients_sequence = None
        self.trainloaders = None
        self.testloaders = None
        self.dataset_name_list = None 
        self.epoch_index = 0 
        self.checkpoint_path = checkpoint_path() + self.args.dataset + '/' + self.args.structure + '/'
        create_if_not_exists(self.checkpoint_path)
        self.net_to_device()

    def net_to_device(self):
        for net in self.nets_list:
            net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def ini(self):
        pass

    def loc_update(self, priloader_list):
        pass

    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                self.nets_list[j].load_state_dict(torch.load(save_path, self.device))
        else:
            pass

    def copy_nets2_prevnets(self):
        for net_id, net in enumerate(self.nets_list):
            self.prev_nets_list[net_id] = copy.deepcopy(net)
    
    def get_params_diff_weights(self):
        weight_dict = {}
        
        for client in self.online_clients:
            client_distance = self.euclidean_distance[client]
            
            delta_weight = (1 - self.args.beta) * (self.previous_delta_weights.get(client, 0)) + self.args.beta * ((client_distance) / sum(self.euclidean_distance.values()))

            new_weight = self.previous_weights.get(client, 1/self.online_num) + delta_weight
            weight_dict[client] = new_weight
            
            self.previous_weights[client] = new_weight
            self.previous_delta_weights[client] = delta_weight
        
        total_weight = sum(weight_dict.values())
        for client in self.online_clients:
            weight_dict[client] /= total_weight
        return weight_dict

    def compute_distance(self, index, update_diff, param_names):
        euclidean_distance = 0
        for key in update_diff:
            if key in param_names:
                euclidean_distance += torch.norm(update_diff[key]).item()

        self.euclidean_distance[index] = euclidean_distance
     
  
    def aggregate_nets(self, freq=None):
        nets_list = self.nets_list

        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        if freq == None and self.args.averaging == 'weight':
            freq = {}
            online_clients_len = {}
            for i in online_clients:
                online_clients_len[i] = len(self.trainloaders[i].sampler)
            online_clients_all = sum(online_clients_len.values())
            for i in online_clients:
                freq[i] = online_clients_len[i] / online_clients_all
        elif freq == None:  
            freq = {}
            online_num = len(online_clients)
            for i in online_clients:
                freq[i] = 1 / online_num
        

        first = True
        for net_id in online_clients:
            net = nets_list[net_id]
            net_para = net.state_dict()
            if first:
                first = False
                for key in net_para:
                    global_w[key] = net_para[key] * freq[net_id]  
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * freq[net_id] 

        print('\t\t'.join(f'{i}:{freq[i]:.3f}' for i in online_clients))
        
        self.global_net.load_state_dict(global_w)
        
        for i in online_clients:
            self.nets_list[i].load_state_dict(global_w)
    
    def aggregate_nets_parameter(self, freq=None):
        
        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        if freq is None and self.args.averaging == 'weight':
            freq = {}
            online_clients_len = {}
            for i in online_clients:
                online_clients_len[i] = len(self.trainloaders[i].sampler)
            online_clients_all = sum(online_clients_len.values())
            for i in online_clients:
                freq[i] = online_clients_len[i] / online_clients_all
        elif freq is None:
            freq = {}
            online_num = len(online_clients)
            for i in online_clients:
                freq[i] = 1 / online_num

        global_params_new = copy.deepcopy(global_w)

        for param_key in global_params_new:
            
            adjusted_weights_list = []
            for client_id in online_clients:
                weight_for_client = freq[client_id] * self.mask_dict[client_id][param_key]
                adjusted_weights_list.append(weight_for_client)
            
            adjusted_weights = torch.stack(adjusted_weights_list).to(self.device)
            
            total_weight_per_param = torch.sum(adjusted_weights, dim=0).unsqueeze(0).to(self.device)
            
            original_weights = torch.Tensor([freq[client_id] for client_id in online_clients]).to(self.device)
            original_weights = original_weights.view(len(online_clients), *[1 for _ in range(len(adjusted_weights.shape)-1)]).expand_as(adjusted_weights)
            weights_for_param = torch.where(total_weight_per_param == 0, original_weights, adjusted_weights / total_weight_per_param)
            
       
            for idx, client_id in enumerate(online_clients):
                update_for_client = self.client_update[client_id][param_key]
                weight_for_client = weights_for_param[idx]
                
                # if "num_batches_tracked" in param_key:
                #     global_params_new[param_key] = global_params_new[param_key] - (update_for_client.float() * weight_for_client).long()
                # else:
                global_params_new[param_key] = global_params_new[param_key] - update_for_client * weight_for_client

        
        self.global_net.load_state_dict(global_params_new)

        
        for i in online_clients:
            self.nets_list[i].load_state_dict(global_params_new)
    
    def consistency_mask(self, client_id, update_diff):
        updates = update_diff 
        if self.epoch_index == 0:
            self.increase_history[client_id] = {key: torch.zeros_like(val) for key, val in updates.items()}
            
            for key in updates:
                self.increase_history[client_id][key] = (updates[key] >= 0).float()
                
            return {key: torch.ones_like(val) for key, val in updates.items()}
        
        mask = {}
        for key in updates:
            positive_consistency = self.increase_history[client_id][key]
            negative_consistency = 1 - self.increase_history[client_id][key]
            
            consistency = torch.where(updates[key] >= 0, positive_consistency, negative_consistency)
            
            mask[key] = (consistency > self.args.threshold).float()
            
        for key in updates:
            increase = (updates[key] >= 0).float()
            self.increase_history[client_id][key] = (self.increase_history[client_id][key] * self.epoch_index + increase) / (self.epoch_index + 1)
            
        return mask
    
    



