import numpy as np
from torch.utils.data import Dataset
from utils import seed_everything
import torch
import pickle
import os


def _generate_solomon_time_windows(depot_xy, node_xy, distribution):
    horizon = distribution.get('tw_horizon', 1.0)
    window_min = distribution.get('tw_window_min', 0.1)
    window_max = distribution.get('tw_window_max', 0.5)
    service_min = distribution.get('service_time_min', 0.01)
    service_max = distribution.get('service_time_max', 0.05)

    batch_size, problem_size, _ = node_xy.shape
    depot_ready = torch.zeros((batch_size, 1), device=node_xy.device)
    depot_due = torch.full((batch_size, 1), horizon, device=node_xy.device)
    depot_service = torch.zeros((batch_size, 1), device=node_xy.device)

    earliest = torch.rand((batch_size, problem_size), device=node_xy.device) * (horizon * 0.5)
    earliest = torch.clamp(earliest, min=0.0)

    window_size = window_min + (window_max - window_min) * torch.rand((batch_size, problem_size), device=node_xy.device)
    latest = torch.min(earliest + window_size, torch.full_like(earliest, horizon))

    ready_time = torch.cat((depot_ready, earliest), dim=1)
    due_time = torch.cat((depot_due, latest), dim=1)
    service_time = torch.cat(
        (
            depot_service,
            service_min + (service_max - service_min) * torch.rand((batch_size, problem_size), device=node_xy.device),
        ),
        dim=1,
    )

    return ready_time, due_time, service_time


def generate_vrp_data(batch_size, problem_size, distribution):
    if distribution['data_type'] == 'uniform':
        depot_xy = torch.rand(size=(batch_size, 1, 2))
        # shape: (batch, 1, 2)
        node_xy = torch.rand(size=(batch_size, problem_size, 2))
        # shape: (batch, problem, 2)

    elif distribution['data_type'] == 'cluster':
        n_cluster = distribution['n_cluster']
        center = np.array([list(np.random.rand(n_cluster * 2)) for _ in range(batch_size)])
        center = distribution['lower'] + (distribution['upper'] - distribution['lower']) * center
        std = distribution['std']
        for j in range(batch_size):
            mean_x, mean_y = center[j, ::2], center[j, 1::2]
            coords = torch.zeros(problem_size + 1, 2)
            for i in range(n_cluster):
                if i < n_cluster - 1:
                    coords[int((problem_size + 1) / n_cluster) * i:int((problem_size + 1) / n_cluster) * (i + 1)] = \
                        torch.cat((torch.FloatTensor(int((problem_size + 1) / n_cluster), 1).normal_(mean_x[i], std),
                                    torch.FloatTensor(int((problem_size + 1) / n_cluster), 1).normal_(mean_y[i], std)),
                                    dim=1)
                elif i == n_cluster - 1:
                    coords[int((problem_size + 1) / n_cluster) * i:] = \
                        torch.cat(
                            (torch.FloatTensor((problem_size + 1) - int((problem_size + 1) / n_cluster) * i, 1).normal_(
                                mean_x[i], std),
                                torch.FloatTensor((problem_size + 1) - int((problem_size + 1) / n_cluster) * i, 1).normal_(
                                    mean_y[i], std)), dim=1)

            coords = torch.where(coords > 1, torch.ones_like(coords), coords)
            coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
            depot_idx = int(np.random.choice(range(coords.shape[0]), 1))
            node_xy = coords[torch.arange(coords.size(0)) != depot_idx].unsqueeze(0) if j == 0 else \
                torch.cat((node_xy, coords[torch.arange(coords.size(0)) != depot_idx].unsqueeze(0)), dim=0)
            depot_xy = coords[depot_idx].unsqueeze(0).unsqueeze(0) if j == 0 else \
                torch.cat((depot_xy, coords[depot_idx].unsqueeze(0).unsqueeze(0)), dim=0)

    elif distribution['data_type'] == 'mixed':
        depot_xy = torch.rand(size=(batch_size, 1, 2))  # shape: (batch, 1, 2)
        n_cluster_mix = distribution['n_cluster_mix']
        center = np.array([list(np.random.rand(n_cluster_mix * 2)) for _ in range(batch_size)])
        center = distribution['lower'] + (distribution['upper'] - distribution['lower']) * center
        std = distribution['std']
        for j in range(batch_size):
            mean_x, mean_y = center[j, ::2], center[j, 1::2]
            mutate_idx = np.random.choice(range(problem_size), int(problem_size / 2), replace=False)
            coords = torch.FloatTensor(problem_size, 2).uniform_(0, 1).to(depot_xy.device)
            for i in range(n_cluster_mix):
                if i < n_cluster_mix - 1:
                    coords[mutate_idx[
                            int(problem_size / n_cluster_mix / 2) * i:int(problem_size / n_cluster_mix / 2) * (i + 1)]] = \
                        torch.cat((torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_x[i], std),
                                    torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_y[i], std)),
                                    dim=1).to(depot_xy.device)
                elif i == n_cluster_mix - 1:
                    coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:]] = \
                        torch.cat((torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i,
                                                        1).normal_(mean_x[i], std),
                                    torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i,
                                                        1).normal_(mean_y[i], std)), dim=1).to(depot_xy.device)

            coords = torch.where(coords > 1, torch.ones_like(coords), coords).to(depot_xy.device)
            coords = torch.where(coords < 0, torch.zeros_like(coords), coords).to(depot_xy.device)
            node_xy = coords.unsqueeze(0) if j == 0 else torch.cat((node_xy, coords.unsqueeze(0)), dim=0)

    # From VRP with RL paper https://arxiv.org/abs/1802.04240
    CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.,
                200: 80.,
                500: 100.,
                1000: 250.
            }
    ready_time, due_time, service_time = _generate_solomon_time_windows(depot_xy, node_xy, distribution)
    data = {
        'loc': node_xy,
        # Uniform 1 - 9, scaled by capacities
        'demand': torch.FloatTensor(torch.randint(1, 10, size=(batch_size, problem_size)).float()) / CAPACITIES[problem_size],
        'depot': depot_xy,
        'ready_time': ready_time,
        'due_time': due_time,
        'service_time': service_time,
    }

    return data

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    ready_time = None
    due_time = None
    service_time = None
    if len(args) > 0:
        if len(args) >= 3:
            depot_types, customer_types, grid_size = args[:3]
        if len(args) >= 6:
            ready_time, due_time, service_time = args[3:6]
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'ready_time': torch.tensor(ready_time, dtype=torch.float) if ready_time is not None else None,
        'due_time': torch.tensor(due_time, dtype=torch.float) if due_time is not None else None,
        'service_time': torch.tensor(service_time, dtype=torch.float) if service_time is not None else None,
    }


class VRPDataset(Dataset):
    
    def __init__(self, filename=None, size=100, num_samples=10000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]

        else:
            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.,
                200: 80.,
                500: 100.,
                1000: 250.
            }
            if distribution == None:
                depot = torch.FloatTensor(num_samples, 2).uniform_(0, 1)
                loc = torch.FloatTensor(num_samples, size, 2).uniform_(0, 1)
                ready_time, due_time, service_time = _generate_solomon_time_windows(
                    depot[:, None, :], loc, distribution={}
                )
                data = {
                        'loc': loc,
                        # Uniform 1 - 9, scaled by capacities
                        'demand': torch.FloatTensor(torch.randint(1, 10, size=(num_samples, size)).float()) / CAPACITIES[size], 
                        'depot': depot,
                        'ready_time': ready_time,
                        'due_time': due_time,
                        'service_time': service_time,
                    }

            else:
                data = generate_vrp_data(num_samples, size, distribution)
                
            self.data = [
                [   
                    data['depot'][i].cpu().numpy(),
                    data['loc'][i].cpu().numpy(),
                    data['demand'][i].cpu().numpy(),
                    1.0,
                    None,
                    None,
                    1.0,
                    data['ready_time'][i].cpu().numpy(),
                    data['due_time'][i].cpu().numpy(),
                    data['service_time'][i].cpu().numpy(),
                ]
                for i in range(data['loc'].shape[0])]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    seed_everything(1234)
    data_size = [1000, 1000, 100]
    problem_size = [100, 200, 500]
    data_type = "cluster" # cluster, mixed, uniform
    distribution = {
        "data_type": data_type,  
        "n_cluster": 3,
        "n_cluster_mix": 1,
        "lower": 0.2,
        "upper": 0.8,
        "std": 0.07,
    }

    for i in range(len(problem_size)):
        val_filename = f'data/vrp{problem_size[i]}_val.pkl'
        test_filename = f'data/vrp_{data_type}{problem_size[i]}_test.pkl'

        # generate validation data
        validation_dataset = VRPDataset(num_samples=data_size[i], size=problem_size[i])
        save_dataset(validation_dataset, val_filename)

        # generate test data
        # test_dataset = VRPDataset(size=problem_size[i], num_samples=data_size[i], distribution=distribution)
        # test_dataset = VRPDataset(size=problem_size[i], num_samples=data_size[i])
        # save_dataset(test_dataset, test_filename)
