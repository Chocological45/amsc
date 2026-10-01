import numpy as np
import ot
import torch
#from plotting import *
from torch.utils.data import DataLoader, SubsetRandomSampler
from scipy.spatial.distance import mahalanobis, cdist, pdist, squareform
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import random
import torch.optim as optim
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from concurrent.futures import ThreadPoolExecutor

class Detect:
    def __init__(
        self,
        reference_num,
        input_dim,
        action_dim,
        num_samples,
        reference=200,
        device='cuda',
        title='',
        num_iter=10,
        one_hot=True,
        normalized=True,
        demo=True,
        embedding_method='lwe',
        swe_num_projections=128,
        swe_num_quantiles=128,
        swe_num_workers=1,
        swe_seed=98,
        swe_normalize_embedding=True,
        reference_seed=98,
    ):
        assert reference is not None, f'Reference not found.'
        self.ref = None
        self.device = device
        self.num_samples = num_samples
        self.num_iter = num_iter
        self.oh = one_hot
        self.normalized = normalized
        self.demo = demo
        self.title = title
        self.input_dim = input_dim
        self.reference_num = reference_num
        self.action_dim = action_dim
        self.embedding_method = str(embedding_method).lower()
        if self.embedding_method not in ('lwe', 'swe'):
            raise ValueError(
                "embedding_method must be either 'lwe' or 'swe', got "
                f"{embedding_method}"
            )
        self.swe_num_projections = int(swe_num_projections)
        self.swe_num_quantiles = int(swe_num_quantiles)
        self.swe_num_workers = max(int(swe_num_workers), 1)
        self.swe_seed = int(swe_seed)
        self.swe_normalize_embedding = bool(swe_normalize_embedding)
        self.reference_seed = int(reference_seed)
        self.swe_directions = None

        self._b = None
        self._a_cache = {}  # batch_size -> a
        
        self.embeddings = []
        self.rewards = []

    def set_input_dim(self, an_input_dim):
        '''A setter method, for manually setting and setting the input dimensionality of the detect
        module.'''
        self.input_dim = an_input_dim

    def get_input_dim(self):
        '''A getter method for accessing the input dimensionality of the detect module.'''
        return self.input_dim

    def set_reference(self, a_task_observation_dim, some_reference_num, some_action_dim):
        '''A setter method, for manually setting and updating the reference for calculating
        the tasks embeddings.'''
        feature_dim = a_task_observation_dim + some_action_dim + 1
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.reference_seed)
        reference = torch.rand(
            some_reference_num,
            feature_dim,
            device=self.device,
            generator=generator,
        )  # Plus one for reward.
        self.ref = reference.float()
        ref_size = self.ref.shape[0]
        self._b = torch.full((ref_size,), 1.0 / ref_size, device=self.device)
        self._set_swe_directions(feature_dim)

    def get_reference(self):
        '''A getter method for accessing the reference which is used to calculate the task
        embeddings'''
        return self.ref

    def _get_a(self, n):
        a = self._a_cache.get(n)
        if a is None:
            a = torch.full((n,), 1.0 / n, device=self.device)
            self._a_cache[n] = a
        return a
        
    def set_num_samples(self, a_num_samples):
        '''A setter method for manually setting the num of samples'''
        self.num_samples = a_num_samples

    def get_num_samples(self):
        '''A getter method for retreiving the detect sample size.'''
        return self.num_samples

    def precalculate_embedding_size(self, a_reference_num, an_inputdim, some_action_dim):
        '''A method for calculating the embedding dimension '''
        if self.embedding_method == 'swe':
            pre_calc_embedding_size = self.swe_num_projections * self.swe_num_quantiles
        else:
            pre_calc_embedding_size = a_reference_num * (an_inputdim + some_action_dim + 1)#Plus one which is the reward.
        return pre_calc_embedding_size

    def _set_swe_directions(self, feature_dim):
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.swe_seed)
        directions = torch.randn(
            self.swe_num_projections,
            int(feature_dim),
            generator=generator,
            dtype=torch.float32,
        )
        directions = F.normalize(directions, dim=1, eps=1e-8)
        self.swe_directions = directions.to(self.device)

    def preprocess_dataset(self, X, action_space_size):
        # X can be numpy or torch; keep it torch
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        # Optional subsample (no DataLoader needed)
        if self.num_samples is not None and X.shape[0] > self.num_samples:
            idx = torch.randperm(X.shape[0], device=X.device)[:self.num_samples]
            X = X.index_select(0, idx)

        X = X.to(self.device).float()

        img = X[:, :self.input_dim]
        act = X[:, self.input_dim:-1]
        reward = X[:, -1:].view(-1, 1)

        if self.normalized:
            # (Your original normalizes globally; keeping that behavior)
            mean = img.mean()
            std = img.std().clamp_min(1e-8)
            img = (img - mean) / std

        if self.oh:
            # assume discrete actions stored as scalar in act
            act = act.view(-1).long()
            act = F.one_hot(act, num_classes=action_space_size).float().to(self.device)

        return torch.cat((img, act, reward), dim=1)

    @torch.no_grad()
    def lwe(self, X, action_space_size, reg=0.05, numItermax=2000):
        if self.embedding_method == 'swe':
            return self.swe(X, action_space_size)

        X = self.preprocess_dataset(X, action_space_size)

        # Ensure ref is on device
        ref = self.ref
        ref_size = ref.shape[0]

        # Cost matrix on GPU (torch.cdist is fast)
        C = ot.dist(X, ref, p=2)

        # torch weights on same device/dtype
        a = torch.full((X.shape[0],), 1.0 / X.shape[0], device=X.device, dtype=X.dtype)
        b = torch.full((ref_size,), 1.0 / ref_size, device=ref.device, dtype=ref.dtype)

        gamma = ot.emd(a, b, C, numItermax=700_000)

        # Barycentric projection-style map (keep your formula)
        f = (ref_size * gamma).T @ X
        f = (f - ref) / (ref_size ** 0.5)

        return f.reshape(-1)  # 1D tensor on GPU

    @torch.no_grad()
    def _swe_chunk_embedding(self, X, directions):
        projections = X @ directions.T
        projections = torch.sort(projections, dim=0).values

        if projections.shape[0] != self.swe_num_quantiles:
            # Interpolate each projection's empirical quantile function to a
            # fixed support size, so embeddings have stable dimensionality.
            projection_channels = projections.T.unsqueeze(0)
            projections = F.interpolate(
                projection_channels,
                size=self.swe_num_quantiles,
                mode='linear',
                align_corners=True,
            ).squeeze(0).T

        return projections.T.reshape(-1)

    @torch.no_grad()
    def swe(self, X, action_space_size):
        X = self.preprocess_dataset(X, action_space_size)

        if self.swe_directions is None or self.swe_directions.shape[1] != X.shape[1]:
            self._set_swe_directions(X.shape[1])

        directions = self.swe_directions
        num_workers = min(self.swe_num_workers, directions.shape[0])
        if num_workers <= 1:
            emb = self._swe_chunk_embedding(X, directions)
        else:
            chunks = [chunk for chunk in torch.chunk(directions, num_workers, dim=0) if chunk.numel() > 0]
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                parts = list(executor.map(lambda chunk: self._swe_chunk_embedding(X, chunk), chunks))
            emb = torch.cat(parts, dim=0)

        emb = emb / max(self.swe_num_projections * self.swe_num_quantiles, 1) ** 0.5
        if self.swe_normalize_embedding:
            emb = F.normalize(emb, dim=0, eps=1e-8)
        return emb
      
    def calculate_lwes_distance(self, lwe1, lwe2):
        '''Calculates the Euclidian Distance of the old vs the new embedding
        It returns a 2D Tensor of size (num * Data-Batch sample size)'''
        eu_dist = (lwe1 - lwe2).pow(2).ravel()
        return eu_dist

    def pwdist(self, tasks_dict):
        '''Computes the Distance of the Ebeddings of Different Taks & create a similarity Matrix for all different Tasks'''
        num_tasks = len(tasks_dict)
        tasks = tasks_dict.values()
        task_ids = list(tasks_dict.keys())

        #initialize pairwise distance matrix
        dist = torch.zeros((num_tasks,num_tasks))

        for k in tqdm(range(int(self.num_iter/2))):
          task_vecs = []
          task_vecs_ = []
          #emb_list = []
          for task in tasks:
            vec = self.lwe(task)
            task_vecs.append(vec)
          for i in range(num_tasks):
            for j in range(i+1,num_tasks):
                dist[i,j]+=torch.linalg.vector_norm(task_vecs[i]-task_vecs[j])/2.
          for task in tasks:
            vec = self.lwe(task)
            task_vecs_.append(vec)
          for i in range(num_tasks):
            dist[i,i]+=torch.linalg.vector_norm(task_vecs[i]-task_vecs_[i])
          for i in range(num_tasks):
            for j in range(i+1,num_tasks):
                dist[i,j]+=torch.linalg.vector_norm(task_vecs_[i]-task_vecs_[j])/2.

    def emb_distance(self, current_embedding, new_calculated_embedding):
        '''Computes the Distance of the newlly calculated embedding and the one that is 
        stored for the current task the agent is solving.'''

        distance  = torch.linalg.vector_norm(current_embedding - new_calculated_embedding)

        return distance
