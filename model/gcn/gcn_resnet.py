import torch
from torch.nn import Module
from torch.nn import Linear
from torch.nn.functional import elu
from typing import Callable, List, Optional, Tuple, Union

try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn.pool import max_pool, max_pool_x, voxel_grid, avg_pool
    from torch_geometric.nn.conv import SplineConv, PointNetConv
    from torch_geometric.nn.norm import BatchNorm
    from torch_geometric.transforms import Cartesian
    print("torch_geometric found")
except ImportError:
    print("torch_geometric not found")


class GraphPooling(Module):
    def __init__(self, size: List[int], transform):
        super(GraphPooling, self).__init__()

        self.transform = transform
        self.voxel_size = list(size)

    def forward(self, data):
        cluster = voxel_grid(data.pos[:, :2], batch=data.batch, size=self.voxel_size)
        data = max_pool(cluster, data=data, transform=self.transform)
        return data
    
class GraphPoolingOut(Module):
    def __init__(self, 
                voxel_size: List[int], 
                size,
                use_time: bool = False):
        super(GraphPoolingOut, self).__init__()

        self.voxel_size = list(voxel_size)
        self.size = size

    def forward(self, data):
        cluster = voxel_grid(data.pos[:, :2], batch=data.batch, size=self.voxel_size)
        x, _ = max_pool_x(cluster, data.x, data.batch, size=self.size)
        return x

class SplineGraphResNet(torch.nn.Module):

    def __init__(self, cfg, num_classes):

        super(SplineGraphResNet, self).__init__()
        dim = cfg.dim
        kernel_size = cfg.kernel_size
        n = cfg.channels
        pooling_outputs = n[7]

        self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
        self.norm1 = BatchNorm(in_channels=n[1])
        self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
        self.norm2 = BatchNorm(in_channels=n[2])

        self.conv3 = SplineConv(n[2], n[3], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
        self.norm3 = BatchNorm(in_channels=n[3])
        self.conv4 = SplineConv(n[3], n[4], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
        self.norm4 = BatchNorm(in_channels=n[4])

        self.conv5 = SplineConv(n[4], n[5], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
        self.norm5 = BatchNorm(in_channels=n[5])
        self.pool5 = GraphPooling((1/8, 1/8), transform=Cartesian(norm=True, cat=False))

        self.conv6 = SplineConv(n[5], n[6], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
        self.norm6 = BatchNorm(in_channels=n[6])
        self.conv7 = SplineConv(n[6], n[7], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
        self.norm7 = BatchNorm(in_channels=n[7])

        self.pool7 = GraphPoolingOut((1/2, 1/2), size=4)
        self.fc = Linear(pooling_outputs * 4, out_features=num_classes, bias=False)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:

        cluster = voxel_grid(data.pos, batch=data.batch, size=[1/16, 1/16, 1/16])
        data = avg_pool(cluster, data=data, transform=Cartesian(norm=True, cat=False))

        data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)
        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)

        x_sc = data.x.clone()
        data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)
        data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        data.x = data.x + x_sc

        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)
        data = self.pool5(data)

        x_sc = data.x.clone()
        data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm6(data.x)
        data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm7(data.x)
        data.x = data.x + x_sc

        x = self.pool7(data)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)

class PNGraphResNet(torch.nn.Module):

    def __init__(self, cfg, num_classes):

        super(PNGraphResNet, self).__init__()
        dim = 3

        kernel_size = 8
        n = [1, 16, 64, 64, 64, 128, 128, 128]
        pooling_outputs = 128

        self.conv1 = PointNetConv(local_nn = Linear(1+3, 16, bias=False), global_nn = Linear(n[1], n[1], bias=False))
        self.norm1 = BatchNorm(in_channels=n[1])

        self.conv2 = PointNetConv(local_nn = Linear(n[1]+3, n[2], bias=False), global_nn = Linear(n[2], n[2], bias=False))
        self.norm2 = BatchNorm(in_channels=n[2])

        self.conv3 = PointNetConv(local_nn = Linear(n[2]+3, n[3], bias=False), global_nn = Linear(n[3], n[3], bias=False))
        self.norm3 = BatchNorm(in_channels=n[3])

        self.conv4 = PointNetConv(local_nn = Linear(n[3]+3, n[4], bias=False), global_nn = Linear(n[4], n[4], bias=False))
        self.norm4 = BatchNorm(in_channels=n[4])

        self.conv5 = PointNetConv(local_nn = Linear(n[4]+3, n[5], bias=False), global_nn = Linear(n[5], n[5], bias=False))
        self.norm5 = BatchNorm(in_channels=n[5])
        self.pool5 = GraphPooling((1/8, 1/8), transform=Cartesian(norm=True, cat=False))

        self.conv6 = PointNetConv(local_nn = Linear(n[5]+3, n[6], bias=False), global_nn = Linear(n[6], n[6], bias=False))
        self.norm6 = BatchNorm(in_channels=n[6])
        self.conv7 = PointNetConv(local_nn = Linear(n[6]+3, n[7], bias=False), global_nn = Linear(n[7], n[7], bias=False))
        self.norm7 = BatchNorm(in_channels=n[7])

        self.pool7 = GraphPoolingOut((1/4, 1/4), size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=num_classes, bias=False)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.x = elu(self.norm1(self.conv1(data.x, data.pos, data.edge_index)))
        data.x = elu(self.norm2(self.conv2(data.x, data.pos, data.edge_index)))

        x_sc = data.x.clone()
        data.x = elu(self.norm3(self.conv3(data.x, data.pos, data.edge_index)))
        data.x = elu(self.norm4(self.conv4(data.x, data.pos, data.edge_index)))

        data.x = data.x + x_sc

        data.x = elu(self.norm5(self.conv5(data.x, data.pos, data.edge_index)))
        data = self.pool5(data)

        x_sc = data.x.clone()

        data.x = elu(self.norm6(self.conv6(data.x, data.pos, data.edge_index)))
        data.x = elu(self.norm7(self.conv7(data.x, data.pos, data.edge_index)))

        data.x = data.x + x_sc

        x = self.pool7(data)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)