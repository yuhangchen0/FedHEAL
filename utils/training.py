import torch
from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
from utils.logger import CsvWriter
from utils.util import generate_online_clients_sequence
from collections import Counter

def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str) -> Tuple[list, list]:
    accs = []
    net = model.global_net
    status = net.training
    net.eval()
    for j, dl in enumerate(test_dl):
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2) if total > 0 else 0
        accs.append(top1acc)
    net.train(status)
    return accs

def local_evaluate(model: FederatedModel, test_dl: DataLoader):
    for client_id in range(model.args.parti_num):
        acc = []
        net = model.nets_list[client_id]
        net.eval()
        for j, dl in enumerate(test_dl):
            correct, total, top1 = 0.0, 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(dl):
                with torch.no_grad():
                    images, labels = images.to(model.device), labels.to(model.device)
                    outputs = net(images)
                    _, max5 = torch.topk(outputs, 5, dim=-1)
                    labels = labels.view(-1, 1)
                    top1 += (labels == max5[:, 0:1]).sum().item()
                    total += labels.size(0)
            top1acc = round(100 * top1 / total, 2) if total > 0 else 0
            acc.append(top1acc)
        print(client_id, acc)
        net.train(True)
        
    

def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)

    model.N_CLASS = private_dataset.N_CLASS
    domains_list = private_dataset.DOMAINS_LIST
    domains_len = len(domains_list)

    if args.rand_dataset:
        max_num = 10
        is_ok = False

        while not is_ok:
            if model.args.dataset == 'fl_officecaltech':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True, p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == 'fl_digits':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)

            result = dict(Counter(selected_domain_list))

            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True

    else:
        if model.args.dataset == 'fl_digits':
            selected_domain_dict = {'mnist': args.mnist, 'usps': args.usps, 'svhn': args.svhn, 'syn': args.syn}
        elif model.args.dataset == 'fl_officecaltech':
            selected_domain_dict = {'caltech': args.caltech, 'amazon': args.amazon, 'webcam': args.webcam, 'dslr': args.dslr}
            
        selected_domain_list = []
        for k in selected_domain_dict:
            domain_num = selected_domain_dict[k]
            for i in range(domain_num):
                selected_domain_list.append(k)

        selected_domain_list = np.random.permutation(selected_domain_list)

        result = Counter(selected_domain_list)
    print(result)
    print(selected_domain_list)
    
    
    model.dataset_name_list = selected_domain_list
    model.online_clients_sequence = generate_online_clients_sequence(args.communication_epoch, args.parti_num, args.online_ratio)
    
    train_loaders, test_loaders = private_dataset.get_data_loaders(selected_domain_list)
    model.trainloaders = train_loaders
    model.testloaders = test_loaders 

    
    if hasattr(model, 'ini'):
        model.ini()
    
    
    accs_dict = {}
    mean_accs_list = []
    
    Epoch = args.communication_epoch
    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index
        print(epoch_index)
        if hasattr(model, 'loc_update'):
            model.loc_update(train_loaders)

        accs = global_evaluate(model, test_loaders, private_dataset.SETTING)
        std = np.std(accs, ddof=1)
        mean_acc = np.mean(accs, axis=0)
        mean_accs_list.append(mean_acc)
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]        
        
        
        combined = [f"{domain}: {acc}" for domain, acc in zip(domains_list, accs)]
        print(f"The {epoch_index} Communcation Round: Method: {model.args.model}")
        print(",\t".join(combined))
        print(f"Mean Accuracy: {mean_acc:.3f}"
            f"\nStandard Deviation: {std:.3f}\n\n")

    if args.csv_log:
        csv_writer.write_acc(accs_dict, mean_accs_list)