import numpy as np
import sklearn.neighbors as knn
import scipy.sparse
import math
import warnings
# from scikits.umfpack import spsolve, splu
from knn import KNN
import cv2 as cv
import copy

import timeit

class Matting:
    def __init__(
        self
    ):
        pass

    def setup(
        self, 
        feature_representation = 'RGBxy_rand',
        n_neighbors = 10,
        _lambda = 100,
        knn_type = 'sklearn',
        metrix_type = 'sparse',
        use_umfpack = True
    ):
        self.feature_representation = feature_representation   # ['RGBxy_rand', 'RGBxy', 'RGB', 'HSVxy', 'HSV']
        self.n_neighbors = n_neighbors                         # 共使用幾個鄰居
        self._lambda = _lambda                                 # 已知 alpha 的限制有多強
        self.knn_type = knn_type                               # [sklearn, hand_craft]
        self.metrix_type = metrix_type                         # [sparse, dense]
        self.use_umfpack = use_umfpack                         # [True, False]

    def preprocess(self, img, trimap):
        trimap = cv.cvtColor(trimap, cv.COLOR_BGR2GRAY)
        trimap = trimap / 255.0

        if self.feature_representation[:3] == 'RGB':
            img = img / 255.0
        elif self.feature_representation[:3] == 'HSV':
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            img = img.astype(float)
            scale_factor = np.array([360.0, 100.0, 100.0])
            img /= scale_factor

        foreground = (trimap == 1.0).astype(int)
        background = (trimap == 0.0).astype(int)

        # M*alpha = V (使用者給定的 alpha, 一種限制)
        assigned_part = foreground + background
        return img, foreground, assigned_part

    def encode_pixel(self, color, i, j, m, n):
        # x,y 座標共用的 normalize factor
        coordinate_factor = math.sqrt(m * m + n * n)

        # cite : https://github.com/dingzeyuli/knn-matting (論文原作者的官方程式 matlab 版本，參考 encode pixel 的想法)
        # 根據文中提到的，添加少量 noise 的在 x,y 座標的資訊上可以減少空間在 feature 中的佔比
        if self.feature_representation == 'RGBxy_rand':
            b, g, r = color
            x_rand = np.random.random() * 1e-6
            y_rand = np.random.random() * 1e-6
            return [r, g, b, i/coordinate_factor + x_rand, j/coordinate_factor + y_rand] # bgr -> rgb (其實並不會影響結果)

        # 不增加 noise 干擾
        elif self.feature_representation == 'RGBxy':
            b, g, r = color
            return [r, g, b, i/coordinate_factor, j/coordinate_factor]
        
        # 與課程提供的教材一致，完全不使用空間的資訊
        elif self.feature_representation == 'RGB':
            b, g, r = color
            return [r, g, b]


        # 與 RGB 概念相同，只是改成 HSV 的版本
        # 根據原作者的說法，HSV 得到的結果並不會比 RGB 好
        elif self.feature_representation == 'HSVxy':
            h, s, v = color
            return [math.cos(h), math.sin(h), s, v, i/coordinate_factor, j/coordinate_factor]
        elif self.feature_representation == 'HSV':
            h, s, v = color
            return [math.cos(h), math.sin(h), s, v]
        else:
            raise NotImplementedError

    def get_img_features(self, img):
        m, n, _ = img.shape
        feature_vectors = []

        # 將每個 pixel encode 成為向量 (encode 的向量形式由 setup 決定)
        for i in range(m):
            for j in range(n):
                feature_vectors.append(self.encode_pixel(img[i][j], i, j, m, n))
        feature_vectors = np.array(feature_vectors)
        return feature_vectors

    def get_pixel_neighbors(self, feature_vectors):
        
        # 使用 sklearn 效能最佳化的 knn (使用 ball tree or kd-tree (c/c++))
        if self.knn_type == 'sklearn':
            model = knn.NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=8).fit(feature_vectors)
            pixel_neighbors = model.kneighbors(feature_vectors)[1] # 找出每個 pixel 最接近的 10 個 pixel (歐式距離)

        # 手寫的 knn，速度差很多 (使用 kd-tree (python))
        # note : 兩個 knn 得到的結果完全一致
        elif self.knn_type == 'hand_craft':
            model = KNN(n_neighbors=self.n_neighbors)
            model.fit(feature_vectors)
            pixel_neighbors = model.get_k_neighbors()
            pixel_neighbors = np.array(pixel_neighbors)
        return pixel_neighbors


    # 計算出 L = (D - A)
    def get_laplacian(self, feature_vectors, pixel_neighbors):
        n_pixels = len(feature_vectors)
        n_neighbors = len(pixel_neighbors[0])
        C = len(feature_vectors[0]) # 表示最小上界

        # 使用稀疏矩陣減少記憶體消耗 (因為絕大多數內容都是 0，而且有對角矩陣)
        if self.metrix_type == 'sparse':
            data = [] # 儲存 kernel function 的數值 (data = || X(i) - X(j) || / C)
            rows = [] # 儲存 row index (稀疏矩陣需要)
            cols = [] # 儲存 column index (稀疏矩陣需要)

            for i in range(n_pixels):
                for j in range(n_neighbors):
                    data.append(1 - np.linalg.norm(feature_vectors[i] - feature_vectors[pixel_neighbors[i][j]]) / C)
                    rows.append(i)
                    cols.append(pixel_neighbors[i][j])

            # 稀疏矩陣 (row, col) 的區域是 data，其餘區域為 0 
            A = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n_pixels, n_pixels))
            D = scipy.sparse.diags(np.ravel(A.sum(axis = 1))) # D 是所有權重的總和，攤平後放到對角線
            L = D - A
        
        # 使用原始矩陣
        else:
            A = np.zeros((n_pixels, n_pixels))
            for i in range(n_pixels):
                for j in range(n_neighbors):
                    A[i][j] = 1 - np.linalg.norm(feature_vectors[i] - feature_vectors[pixel_neighbors[i][j]]) / C

            D = np.diag(np.ravel(A.sum(axis = 1)))
            L = D - A
        return L

    def solve(self, H, V):
        warnings.filterwarnings('error')
        try:
            # cite : https://github.com/scikit-umfpack/scikit-umfpack/blob/master/scikits/umfpack/interface.py#L93 (實際做法)
            # 使用 use_umfpack 表示要不要先做 LU factorization 加速 (Pr * (R^-1) * A * Pc = L * U)
            # 這會大幅度的提昇計算速度
            alpha = scipy.sparse.linalg.spsolve(H, V, use_umfpack=True)
        except Warning:
            # 陣列可能不存在反矩陣
            # |b - A x| is minimized.
            alpha = scipy.sparse.linalg.lsqr(H, V)[0]

        # alpha 的數值應該在 [0, 1] 之間
        alpha = np.clip(alpha, 0, 1)
        return alpha

    def knn_matting(self, img, trimap):
        m, n, _ = img.shape
        img, foreground, assigned_part = self.preprocess(img, trimap)
        feature_vectors = self.get_img_features(img)
        pixel_neighbors = self.get_pixel_neighbors(feature_vectors)
        L = self.get_laplacian(feature_vectors, pixel_neighbors)
        
        # M 表示該 pixel 有沒有被使用者標記，assigned_part 是有被標記的部份 (foreground, background)
        M = scipy.sparse.diags(np.ravel(assigned_part))

        # V 表示被 assign 的 alpha，前景為 1
        V = self._lambda * np.transpose(np.ravel(foreground))

        # alpha 前的係數
        H = (L + self._lambda * M)

        # 解決 a @ x == b 的問題，其中 x 是未知數 (alpha)
        alpha = self.solve(H, V)
        alpha = alpha.reshape(m, n) # 把 1 維的 alpha 重新拉回 mxn
        return alpha

    def combine_image(self, object, scene, alpha):
        object_height, object_width, _ = object.shape
        scene_height, scene_width, _ = scene.shape
        new_height, new_width = int(1.5*object_height), int(1.5*scene_width*(object_height/scene_height))
        h_offset, w_offset = new_height-object_height, int(0.25*object_width)

        scene = cv.resize(scene, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        result_image = copy.deepcopy(scene)

        for i in range(object_height):
            for j in range(object_width):
                result_image[i+h_offset][j+w_offset] = \
                    alpha[i][j] * object[i][j] + (1 - alpha[i][j]) * result_image[i+h_offset][j+w_offset]
        
        return result_image
    