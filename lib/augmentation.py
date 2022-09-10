from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from lib.graph_functional import *



class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    def unfold(self):
        # return: Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights


class Augmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph):
        # return:  -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None
    ): 
    # return: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(Graph(x, edge_index, edge_weight)).unfold()


class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: Graph):
        # return:  -> Graph:
        for aug in self.augmentors:
            g = aug.augment(g)
        return g


class RandomChoice(Augmentor):
    def __init__(self, augmentors: List[Augmentor], num_choices: int):
        super(RandomChoice, self).__init__()
        assert num_choices <= len(augmentors)
        self.augmentors = augmentors
        self.num_choices = num_choices

    def augment(self, g: Graph):
        # return  -> Graph:
        num_augmentors = len(self.augmentors)
        perm = torch.randperm(num_augmentors)
        idx = perm[:self.num_choices]
        for i in idx:
            aug = self.augmentors[i]
            g = aug.augment(g)
        return g

# indentity
class Identity(Augmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, g: Graph):
        return g

# node_dropping

class NodeDropping(Augmentor):
    def __init__(self, pn: float):
        super(NodeDropping, self).__init__()
        self.pn = pn

    def augment(self, g: Graph):
    #  -> Graph:
        x, edge_index, edge_weights = g.unfold()

        edge_index, edge_weights = drop_node(edge_index, edge_weights, keep_prob=1. - self.pn)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

# node_shuffling
class NodeShuffling(Augmentor):
    def __init__(self):
        super(NodeShuffling, self).__init__()

    def augment(self, g: Graph):
        x, edge_index, edge_weights = g.unfold()
        x = permute(x)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

#  node feature_dropout
class FeatureDropout(Augmentor):
    def __init__(self, pf: float):
        super(FeatureDropout, self).__init__()
        self.pf = pf

    def augment(self, g: Graph):
        x, edge_index, edge_weights = g.unfold()
        x = dropout_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

# feature_masking
class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph):
        x, edge_index, edge_weights = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

# ppr_diffusion
class PPRDiffusion(Augmentor):
    def __init__(self, alpha: float = 0.2, eps: float = 1e-4, use_cache: bool = True, add_self_loop: bool = True):
        super(PPRDiffusion, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop

    def augment(self, g: Graph):
        if self._cache is not None and self.use_cache:
            return self._cache
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = compute_ppr(
            edge_index, edge_weights,
            alpha=self.alpha, eps=self.eps, ignore_edge_attr=False, add_self_loop=self.add_self_loop
        )
        res = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        self._cache = res
        return res

# rw_sampling
class RWSampling(Augmentor):
    def __init__(self, num_seeds=1000, walk_length=10):
        super(RWSampling, self).__init__()
        self.num_seeds = num_seeds
        self.walk_length = walk_length

    def augment(self, g: Graph):
        x, edge_index, edge_weights = g.unfold()

        edge_index, edge_weights = random_walk_subgraph(edge_index, edge_weights, batch_size=self.num_seeds, length=self.walk_length)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

# markov_diffusion
class MarkovDiffusion(Augmentor):
    def __init__(self, alpha: float = 0.05, order: int = 16, sp_eps: float = 1e-4, use_cache: bool = True,
                 add_self_loop: bool = True):
        super(MarkovDiffusion, self).__init__()
        self.alpha = alpha
        self.order = order
        self.sp_eps = sp_eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop

    def augment(self, g: Graph):
        if self._cache is not None and self.use_cache:
            return self._cache
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = compute_markov_diffusion(
            edge_index, edge_weights,
            alpha=self.alpha, degree=self.order,
            sp_eps=self.sp_eps, add_self_loop=self.add_self_loop
        )
        res = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        self._cache = res
        return res

# edge_adding
class EdgeAdding(Augmentor):
    def __init__(self, pe: float):
        super(EdgeAdding, self).__init__()
        self.pe = pe

    def augment(self, g: Graph):
        # return: -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index = add_edge(edge_index, ratio=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


# edge_attr_masking
class EdgeAttrMasking(Augmentor):
    def __init__(self, pf: float):
        super(EdgeAttrMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph):
        x, edge_index, edge_weights = g.unfold()
        if edge_weights is not None:
            edge_weights = drop_feature(edge_weights, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

# edge_removing
class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, g: Graph):
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = dropout_adj(edge_index, edge_attr=edge_weights, p=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)




def generate_aug(args_aug, p=0.1, num_seeds=1000, walk_length=10):
    if args_aug == 'indentity':
        aug = Identity()
    elif args_aug == 'EdgeAdding':
        aug = EdgeAdding(p)
    elif args_aug == 'EdgeRemoving':
        aug = EdgeRemoving(p)
    elif args_aug == 'FeatureMasking':
        aug = FeatureMasking(p)
    elif args_aug == 'FeatureDropout':
        aug = FeatureDropout(p)
    elif args_aug == 'EdgeAttributeMasking':
        aug = EdgeAttrMasking(p)
    elif args_aug == 'PersonalizedPageRank':
        aug = PPRDiffusion()
    elif args_aug == 'MarkovDiffusionKernel':
        aug = MarkovDiffusion()
    elif args_aug == 'NodeDropping':
        aug = NodeDropping(p)
    elif args_aug == 'NodeShuffling':
        aug = NodeShuffling()
    elif args_aug == 'RWS':
        aug = RWSampling(num_seeds, walk_length)
    else:
        print('Invalid Augmentation Method!')
        raise 
    return aug




if __name__ == '__main__':
    dataset = TUDataset('/home/sjw29823/DGCL/data', name='PTC_MR')

    aug = EdgeAdding(0.1)
    # aug = Identity()
    # aug = NodeDropping(0.2)
    # print(dataset[300].edge_index.size())
    aug = NodeDropping(0.5)
    aug = NodeShuffling()
    aug = FeatureDropout(0.3)
    aug = FeatureMasking(0.7)
    g = aug(dataset[300].x, dataset[300].edge_index)
    x2, edge_index2, edge_weight2 = g
    print(dataset[300].edge_index, dataset[300].x)
    
    print(edge_index2, x2)


    # print(aug(dataset[0]))







