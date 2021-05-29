import numpy as np

########################## simulations - BT #############################
def get_beta_with_gap(N, delta = 0.1):
    beta = np.arange(N) * delta
    beta = beta - np.mean(beta)
    return beta

def get_game_matrix(beta, edge_list, total = 1):
    N = len(beta)
    game_matrix = np.zeros((N,N))
    for edge in edge_list:
        i, j = edge
        pij = np.exp(beta[i] - beta[j]) / (1 + np.exp(beta[i] - beta[j]))
        nij = np.random.binomial(n = total, p = pij, size = 1)
        game_matrix[i,j], game_matrix[j,i] = nij, total - nij
    return game_matrix


########################## simulations - comparison graph #############################

#### Barbell with random bridge

def get_complete_adj(n = 50):
    return np.ones((n,n)) - np.eye(n)

def random_edge(n1, n2, edge):
    index = np.random.choice(n1 * n2, edge, replace = False)
    i1 = np.repeat(np.arange(n1), n2)[index]
    i2 = np.tile(np.arange(n1), n2)[index]    
    A = np.zeros((n1, n2))
    A[i1,i2] = 1
    return A

def get_barbell_adj(n1 = 50, n2 = 50, bridge = 1):
    n = n1 + n2
    A = np.zeros((n,n))
    A[:n1,:n1] = get_complete_adj(n = n1)
    A[n1:,n1:] = get_complete_adj(n = n2)
    A[:n1,n1:] = random_edge(n1 = n1, n2 = n2, edge = bridge)
    A[n1:,:n1] = A[:n1,n1:].copy().T
    return A

def get_barbell_edge(n1 = 50, n2 = 50, bridge = 1):
    n = n1 + n2
    A = get_barbell_adj(n1 = n1, n2 = n2, bridge = bridge)
    edge = get_edge(A)
    return edge

def get_edge(A, list = False):
    edge = np.where(np.triu(A))
    if list:
        return [(x,y) for x, y in zip(edge[0],edge[1])]
    else:
        return zip(edge[0], edge[1])


#### Island graph

def get_3island_adj(n = 100, n1 = 30, n2 = 70, n3 = 30, k = None):
    if k is None:
        k = n1 // 2
    A = np.zeros((n,n))
    A[:n1,:n1] = get_complete_adj(n = n1)
    A[-n3:,-n3:] = get_complete_adj(n = n3)
    A[k:k+n2,k:k+n2] = get_complete_adj(n = n2)
    return A

def get_island_adj(ni = 30, no = 5, k = 5, n = None):
    if n == None:
        n = ni * k - no * (k - 1)
    else:
        k = n // (ni - no) + 1
    A = np.zeros((n,n))
    for i in range(k):
        s = (ni - no) * i
        e = min(n, s + ni)
        A[s:e,s:e] = get_complete_adj(n = e - s)
    return A


#### k-Cayley graph

def get_k_cayley_adj(n, k, permute = False):
    # k-cayley graph
    adj = np.zeros((n,n))
    for i in range(n):
        for j in range(1, k + 1):
            col = (i + j) % n
            adj[i, col] = 1
            
            col = (i - j) % n
            adj[i, col] = 1
    if permute:
        P = np.zeros((n,n))
        P[np.arange(n), np.random.permutation(n)] = 1
        adj = P.T @ adj @ P
    return adj


#### Topological quantities

def lambda2_A(A):
    n = A.shape[0]
    La = np.diag(np.sum(A, axis = 0)) - A
    lambdas = np.linalg.eigvalsh(La)
    return lambdas[1], lambdas[-1]

def min_n_ij(A):
    return np.min(A @ A)

def get_topo(A):
    lam2, lamn = lambda2_A(A)
    n_ij = min_n_ij(A)
    deg = np.sum(A, axis = 0)
    n_max, n_min = np.max(deg), np.min(deg)
    return lam2, n_ij, n_max, n_min




########################## estimators #############################

def get_mle_reg(game_matrix, rho = None, method = 'newton'):
    n, _ = game_matrix.shape
    total = np.max(game_matrix + game_matrix.T)
    
    if not rho:
        rho = (n * np.log(n) / total)**0.5
    
    if method == 'newton':
        beta_hat = mle.newton_l2_sq(game_matrix, l_penalty = rho)[1].reshape(-1)
    
    beta_hat -= np.mean(beta_hat)
    
    return beta_hat


########################## metrics #############################

def get_bound_yan(kappa, n_ij, n, L, no_kappa = False):
    if no_kappa:
        return 1 / n_ij * np.sqrt(L * n * np.log(n + 1))
    else:
        return np.exp(kappa) / n_ij * np.sqrt(L * n * np.log(n + 1))
    
def get_bound_shah(kappa, lam2, n, L, no_kappa = False):
    if no_kappa:
        return np.sqrt(1/lam2 * (n * np.log(n) / L))
    else:
        return np.exp(4 * kappa) * np.sqrt(1/lam2 * (n * np.log(n) / L))

def get_bound_our(kappa, lam2, n_max, n_min, n, L, no_kappa = False):
    if no_kappa:
        return 1/lam2 * (n_max/n_min) * np.sqrt(n / L) + 1/lam2 * np.sqrt(n_max * np.log(n) / L)
    else:
        return np.exp(2 * kappa)/lam2 * (n_max/n_min) * np.sqrt(n / L) + np.exp(kappa)/lam2 * np.sqrt(n_max * np.log(n) / L)



def l_infty_error(v, v_hat):
    return np.max(np.abs(v - v_hat))

def ranking_error(v,v_hat):
    r = stats.rankdata(v, method = 'min')
    r_hat = stats.rankdata(v_hat, method = 'min')
    return np.sum(np.abs(r - r_hat))

def topk_error(v, v_hat, k = None):
    n = len(v)
    if not k:
        k = n // 3
    r = stats.rankdata(v, method = 'min')
    r_hat = stats.rankdata(v_hat, method = 'min')
    
    ind = np.arange(1, n + 1)
    top = np.where(r <= k)[0]
    bottom = np.setdiff1d(ind, top)
    top_hat = np.where(r_hat <= k)[0]
    bottom_hat = np.setdiff1d(ind, top_hat)
    
    diff = len(np.setdiff1d(top,top_hat)) + len(np.setdiff1d(bottom,bottom_hat))
    diff /= (2 * k)
    return diff

def get_error(v, v_hat, k = None):
    l_infty = l_infty_error(v, v_hat)
    rank = ranking_error(v, v_hat)
    topk = topk_error(v, v_hat, k = k)
    return l_infty, rank, topk