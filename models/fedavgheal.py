import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedAvGHEAL(FederatedModel):
    NAME = 'fedavgheal'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedAvGHEAL, self).__init__(nets_list,args,transform)
        
        self.client_update = {}
        self.increase_history = None
        self.mask_dict = {}
        
        self.euclidean_distance = {}
        self.previous_weights = {}
        self.previous_delta_weights = {}
        

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

 
    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(self.online_clients)

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
            
            if self.args.wHEAL == 1:
                net_params = self.nets_list[i].state_dict()
                global_params = self.global_net.state_dict()
                param_names = [name for name, _ in self.nets_list[i].named_parameters()]
                update_diff = {key: global_params[key] - net_params[key] for key in global_params}
                
                mask = self.consistency_mask(i, update_diff)
                self.mask_dict[i] = mask
                masked_update = {key: update_diff[key] * mask[key] for key in update_diff} 
                self.client_update[i] = masked_update
                    
                self.compute_distance(i, self.client_update[i], param_names)
                
        freq = self.get_params_diff_weights()
        self.aggregate_nets_parameter(freq)


    
    def _train_net(self,index,net,train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index,loss)
                optimizer.step()
        



