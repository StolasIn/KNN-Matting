from tqdm import tqdm
import heapq
import numpy as np
from multiprocessing import Pool
import os

class Node:
    def __init__(self, feature=None, index=None, left=None, right=None, split=None):
        self.feature = feature
        self.index = index
        self.left = left
        self.right = right
        self.split = split

class KDTree:
    def __init__(self, data, index):
        self.n_dim = data.shape[1]

        # [data, index]
        data = np.hstack([data, np.array(index).reshape(len(index), 1)])
        self.root = self.build(data, 0)
        self.result = []

    def build(self, data, depth):
        n_node = len(data)
        if n_node == 0:
            return None
        
        axis = depth % (self.n_dim) # 要拿來切的 dim
        mid = n_node // 2           # 中位數的 index
        data = sorted(data, key = lambda item: item[axis])

        node = Node(data[mid][:-1], data[mid][-1], split = axis)

        # 把資料切成左子樹右子樹
        node.left = self.build(data[:mid], depth + 1)
        node.right = self.build(data[mid + 1:], depth + 1)
        return node
    
    def _search(self, node, target, k):
        if node is None:
            return
        
        feature = node.feature
        distance = np.sqrt(np.sum(np.square(target - feature)))

        # 維持前 k 近的鄰居，heapq 只能用 max-heap 所以需要加負號
        if len(self.result) < k:
            heapq.heappush(self.result, (-distance, node.index))
        elif -distance > self.result[0][0]:
            heapq.heapreplace(self.result, (-distance, node.index))

        # 根據切分軸考慮要往哪邊走
        axis = node.split
        if abs(target[axis] - feature[axis]) < -self.result[0][0] or len(self.result) < k:
            self._search(node.left, target, k)
            self._search(node.right, target, k)
        elif target[axis] < feature[axis]:
            self._search(node.left, target, k)
        else:
            self._search(node.right, target, k)

    def search(self, target, k):
        self.result = []
        self._search(self.root, target, k)
        return sorted(self.result, key=lambda x: -x[0])

class KNN:
    def __init__(
        self,
        n_neighbors
    ):
        self.n_neighbors = n_neighbors
        self.tree = None

    def fit(self, feature_vectors):
        index = range(len(feature_vectors))
        self.tree = KDTree(feature_vectors, index)
        self.data = feature_vectors
        self.data_len = len(feature_vectors)
    
    def predict(self, feature):
        result = self.tree.search(feature, self.n_neighbors)
        result = [int(item[1]) for item in result]
        return result
    
    def get_k_neighbors(self):
        results = []
        for item in tqdm(self.data):
            results.append(self.predict(item))
        return results
    
    # multi-processing
    def get_k_neighbors_mp(self):
        pool = Pool(os.cpu_count())
        results = pool.map(self.predict, self.data)
        return results