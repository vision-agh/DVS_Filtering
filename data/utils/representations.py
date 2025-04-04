import torch
import numpy as np
import torchvision.transforms as transforms

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import Cartesian

def generate_event_frame(events, cfg):
    width, height = cfg.general.org_dim
    cfg = cfg.representation.event_frame

    W, H = cfg.dim

    x = events[:, 0].long()
    y = events[:, 1].long()
    p = events[:, 2].long()

    frame = torch.zeros(2, height, width, dtype=torch.float32)

    indices = torch.stack([ p, 
                            y, 
                            x], dim=0)

    values = torch.ones_like(events[:, 0], dtype=torch.float32)
    frame.index_put_(tuple(indices), values, accumulate=True)
    frame = transforms.Resize((H, W))(frame)
    return frame

def generate_event_voxel(events, cfg):
    width, height = cfg.general.org_dim
    time_window = cfg.general.time_window

    cfg = cfg.representation.event_voxel
    T = cfg.T
    W, H = cfg.dim

    x = events[:, 0].long()
    y = events[:, 1].long()
    p = events[:, 2].long()

    if x.numel() == 0:
        return torch.zeros(2*T, H, W, dtype=torch.float32)
    
    t = events[:, 3].long() - events[:, 3].min()
    t = t / time_window
    t = t * T
    t = torch.floor(t).long()
    t = torch.clamp(t, min=0, max=T-1)

    voxel = torch.zeros(2, T, height, width, dtype=torch.float32)
    indices = torch.stack([ p, 
                            t,
                            y, 
                            x], dim=0)

    values = torch.ones_like(events[:, 0], dtype=torch.float32)
    voxel.index_put_(tuple(indices), values, accumulate=True)

    voxel = voxel.reshape(-1, height, width)
    voxel = transforms.Resize((H, W))(voxel)
    return voxel


def generate_event_spikes(events, cfg):
    width, height = cfg.general.org_dim
    time_window = cfg.general.time_window

    cfg = cfg.representation.event_spikes
    T = cfg.T
    W, H = cfg.dim

    x = events[:, 0].long()
    y = events[:, 1].long()
    p = events[:, 2].long()

    if x.numel() == 0:
        print("No events")
        return torch.zeros(T, 2, H, W, dtype=torch.float32)
    
    # Normalize time from 50000 to T
    t = events[:, 3].long() - events[:, 3].min()
    t = t / time_window
    t = t * T
    t = torch.floor(t).long()

    voxel = torch.zeros(T, 2, height, width, dtype=torch.float32)

    indices = torch.stack([ t, 
                            p,
                            y, 
                            x], dim=0)

    values = torch.ones_like(events[:, 0], dtype=torch.float32)
    voxel.index_put_(tuple(indices), values, accumulate=True)
    voxel = transforms.Resize((H, W))(voxel)
    return voxel

def generate_event_graph(events, cfg):
    time_window = cfg.general.time_window
    width, height = cfg.general.org_dim

    cfg = cfg.representation.event_graph
    W, H = cfg.dim 

    x = (events[:, 0] / width) * W
    y = (events[:, 1] / height) * H

    if x.numel() == 0:
        return Data(x=torch.zeros(1, 1), 
                    pos=torch.zeros(1, 3), 
                    edge_index=torch.zeros(2, 1, dtype=torch.long), 
                    edge_attr=torch.zeros(1, 3))

    p = events[:, 2] 
    t = events[:, 3] - events[:, 3].min()
    t = t / time_window

    p = p.unsqueeze(1)

    pos = torch.stack([x, y, t], dim=1)

    graph = Data(pos=pos, x=p)
    graph.edge_index = radius_graph(graph.pos, r=cfg.radius, max_num_neighbors=cfg.max_num_neighbors)

    edge_attr = Cartesian(norm=True, cat=False, max_value=cfg.radius)
    graph = edge_attr(graph)

    return graph
