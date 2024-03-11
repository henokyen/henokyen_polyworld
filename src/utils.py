import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon,LineString, Point
import math

def create_polygon(seg):
    points = [[seg[i*2], seg[i*2+1]] for i in range(len(seg) // 2)]
    return Polygon(points)

def angle_between_points(A, B, C, batch=False):
    d = 2 if batch else 1
    AB = A - B
    BC = C - B
    epsilon = 3e-8
    
    AB_mag = torch.norm(AB, dim=d) + epsilon
    BC_mag = torch.norm(BC, dim=d) + epsilon
    
    dot_product = torch.sum(AB * BC, dim=d)
    cos_theta = dot_product / (AB_mag * BC_mag)
    
    zero_mask = (AB_mag == 0) | (BC_mag == 0)
    cos_theta[zero_mask] = 0  
    theta = torch.acos(torch.clamp(cos_theta, -1 + epsilon, 1 - epsilon))
    theta[zero_mask] = 0
    return theta * 180 / math.pi

def sinkhorn_logsumexp(cost_matrix, reg=1e-1, maxiter=100):
      """
      https://gist.github.com/louity/0629e2d12ff4ac96573b3a541246e162
      Log domain version on sinkhorn distance algorithm ( https://arxiv.org/abs/1306.0895 ).
      Inspired by https://github.com/gpeyre/SinkhornAutoDiff/blob/master/sinkhorn_pointcloud.py ."""
      b, n, n = cost_matrix.shape                                                    

      mu = torch.FloatTensor(n).fill_(1./n).log().expand(b, -1)                                                                          
      nu = torch.FloatTensor(n).fill_(1./n).log().expand(b, -1)                                     

      if torch.cuda.is_available():                                        
        mu, nu = mu.cuda(), nu.cuda()                                    

      def M(u, v):                                                                           
        return (-cost_matrix + u.unsqueeze(2) + v.unsqueeze(1)) / reg    

      u, v = 0. * mu, 0. * nu                                                  

      # Actual Sinkhorn loop                                                     
      for i in range(maxiter):                                                   

        u = reg * (torch.log(mu) - torch.logsumexp(cost_matrix+v.unsqueeze(1) , dim=2))       
        v = reg * (torch.log(nu) - torch.logsumexp(cost_matrix+u.unsqueeze(2), dim=1))

      return torch.exp(M(u, v))

def log_optimal_transport_batch(Z, iters):
    """
    Computes the optimal transport between all pairs of rows and columns of a batch of cost matrices Z in log space,
    using the Sinkhorn algorithm.

    Args:
        Z: a tensor of shape (batch_size, m, n) representing a batch of cost matrices, where m is the number of rows
           and n is the number of columns.
        iters: the number of Sinkhorn iterations to perform.

    Returns:
        A tensor of the same shape as Z, containing the optimal transport plan between all pairs of rows and columns
        in the batch.
    """
    batch_size, m, n = Z.shape
    log_mu = -torch.tensor(m).to(Z).log().expand(batch_size, m)
    log_nu = -torch.tensor(n).to(Z).log().expand(batch_size, n)
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)

    for _ in range(iters):
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)

    return Z + u.unsqueeze(-1) + v.unsqueeze(-2)


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    alpha = torch.tensor(1.0)
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)
    
    
    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)
    
    if torch.cuda.is_available(): 
        bins0 = bins0.cuda()
        bins1 = bins1.cuda()
        alpha = alpha.cuda()
    

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
    
    if torch.cuda.is_available():                                        
        log_mu, log_nu = log_mu.cuda(), log_nu.cuda()

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z[:,:-1,:-1]

def log_optimal_transport_2(scores: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    norm = - (ms + ns).log()
    log_mu,log_nu = norm.expand(m).expand(b, -1),norm.expand(n).expand(b, -1)
    
    if torch.cuda.is_available():                                        
        log_mu, log_nu = log_mu.cuda(), log_nu.cuda()
  
    Z = log_sinkhorn_iterations(scores, log_mu, log_nu, iters)
    Z = Z - norm  #multiply probabilities by M+N
    return Z

def log_optimal_transport_3(scores: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
   
    log_mu = -torch.tensor(m).to(scores).log().expand([m]).expand(b,-1)
    log_nu = -torch.tensor(n).to(scores).log().expand([n]).expand(b,-1)
    
    if torch.cuda.is_available():                                        
        log_mu, log_nu = log_mu.cuda(), log_nu.cuda()
  
    Z = log_sinkhorn_iterations(scores, log_mu, log_nu, iters)
    return Z

def scores_to_permutations(scores):
    """
    Input a batched array of scores and returns the hungarian optimized 
    permutation matrices.
    """
    B, N, N = scores.shape

    scores = scores.detach().cpu().numpy()
    perm = np.zeros_like(scores)
    for b in range(B):
        r, c = linear_sum_assignment(-scores[b])
        perm[b,r,c] = 1
    return torch.tensor(perm)

def remove_diagonal(indices,nms_graph, N):
    indx_remove = []
    for indx in torch.arange(N):
        p_original = tuple(nms_graph[indx].tolist())
        if indices[p_original]:
            indx_remove.append(indx.item())
    return indx_remove

def bubble_merge(poly):
    s = 0
    P = len(poly)
    while s < P:
        head = poly[s][-1]

        t = s+1
        while t < P:
            tail = poly[t][0]
            if head == tail:
                poly[s] = poly[s] + poly[t][1:]
                del poly[t]
                poly = bubble_merge(poly)
                P = len(poly)
            t += 1
        s += 1
    return poly
    

def permutations_to_polygons(perm, graph, out='torch'):
    B, N, N = perm.shape
    diag = torch.logical_not(perm[:,range(N),range(N)])
    batch = []
    for b in range(B):
        b_perm = perm[b]
        b_graph = graph[b]
        b_diag = diag[b]
        
        idx = torch.arange(N)[b_diag]
        
        if idx.shape[0] > 0:
            
            # If there are vertices in the batch

            b_perm = b_perm[idx,:]
            b_graph = b_graph[idx,:]
            b_perm = b_perm[:,idx]

            first = torch.arange(idx.shape[0]).unsqueeze(1)
            second = torch.argmax(b_perm, dim=1).unsqueeze(1).cpu()

            polygons_idx = torch.cat((first, second), dim=1).tolist()
            polygons_idx = bubble_merge(polygons_idx)

            batch_poly = []
            for p_idx in polygons_idx:
                if out == 'torch':
                    batch_poly.append(b_graph[p_idx,:])
                elif out == 'numpy':
                    batch_poly.append(b_graph[p_idx,:].numpy())
                elif out == 'list':
                    g = b_graph[p_idx,:] * 320 / 320
                    g[:,0] = -g[:,0]
                    g = torch.fliplr(g)
                    batch_poly.append(g.tolist())
                elif out == 'coco':
                    g = b_graph[p_idx,:] * 320 / 320
                    g = torch.fliplr(g)
                    batch_poly.append(g.view(-1).tolist())
                else:
                    print("Indicate a valid output polygon format")
                    exit()
        
            batch.append(batch_poly)

        else:
            # If the batch has no vertices
            batch.append([])

    return batch


def permutation_mask(segs,p_size):
    perm = np.zeros((p_size, p_size))
    d_index = {}
    i = 0
    for ann in segs:
        #p = ann['segmentation'][0]
        p = [round(i) for i in ann['segmentation'][0]]
        point_pair = list(zip(p[::2], p[1::2]))
        for (x,y) in point_pair:
            if (x,y) not in d_index:
                d_index[(x,y)]= i
                i+=1
       
        # filling permutation matrix for each polygon 
        n = len(p) //2
        for c in range(n-1):
            x_index = d_index[(point_pair[c])]
            y_index = d_index[(point_pair[c+1])]
            perm[x_index][y_index] = 1
    return perm, d_index

#sorting N=256 points 
class Node:
    def __init__(self, point ):
        self.point = point 
        self.left = None
        self.right = None
class KDTree:
    def __init__(self, data):
        self.data = data
        self.dim = len(data)
        self.k = 2
        self.closest = []
        self.root = self.build_kdtree()

    def build_kdtree(self):
        root = None
        for point in self.data:
            root = self.insert(root, point)
        return root
    
    def insert(self, root, point):
        return self.inserrec(root, point, 0)
    
    def inserrec(self, root, point, depth):
        if not root:
            return Node(point)

        cd = depth % self.k
        
        if point[cd] < root.point[cd]:
            root.left = self.inserrec(root.left, point, depth+1)
        else:
            root.right = self.inserrec(root.right, point, depth+1)
        
        return root

    def euclidean_distance(self, source, target):
        return math.sqrt(sum([((target[i] - source[i])**2) for i in range(self.k)]))
    
    def search_KNN(self, target , K=1):
        self.closest = []
        self.nearest_point(self.root, target, 0)

    def nearest_point(self,root, target, depth):
        if not root:
            return 
        curr_point = root.point
        
        curr_distance = self.euclidean_distance(curr_point, target)
        cd = depth % self.k
        if not self.closest:
            self.closest.append((root,curr_distance))
        elif curr_distance < self.closest[0][1]:
            self.closest = self.closest[1:] + [(root, curr_distance)]
        
        cd = depth % self.k
        if abs(target[cd] - curr_point[cd]) < self.closest[0][1]:
            self.nearest_point(root.left, target, depth +1)
            self.nearest_point(root.right, target, depth +1)
        elif target[cd] < curr_point[cd]:
            self.nearest_point(root.left, target, depth +1)
        else:
            self.nearest_point(root.right, target, depth +1)

'''
print(indices[0])
print()
indx_remove = []
for indx in torch.arange(N):
    #p_updated=tuple(graph[0][indx].tolist())
    p_original = tuple(nms_graph[0][indx].tolist())
    #print(p_original ,'->', p_updated)
    if indices[0][p_original]:
        indx_remove.append(indx.item())
print (indx_remove)  
print(len(indx_remove))

#idx = torch.arange(N)[b_diag]
#print(idx)
res_indx = filter(lambda i: i not in indx_remove, torch.arange(N))
idx = [tensor.item() for tensor in res_indx]
print(idx)
idx = torch.from_numpy(np.asarray(idx))
idx

'''

'''
def nearest_neighboor_2d(root, data, gt_points):
    cndd_gt = {(k[0], k[1]):-1 for k in data}
    gt_cndd = {}
    for gt_pt in gt_points:
        for (x,y) in zip(gt_pt[::2], gt_pt[1::2]):
            root.search_KNN(target= (x,y))
            #print('nearest neigboor of {} is {} {}'.format( (x,y), root.closest[0][0].point,gt_index[(x,y)]))
            cndd_gt[tuple(root.closest[0][0].point)] = (x,y)
            gt_cndd[(x,y)] = root.closest[0][0].point
    return cndd_gt, gt_cndd
'''