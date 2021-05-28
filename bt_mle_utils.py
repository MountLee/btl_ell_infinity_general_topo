import sys
import numpy as np
import scipy as sc
import pandas as pd
import scipy.linalg as spl

def neg_log_like(beta,game_matrix):
    '''
    calculate the negative loglikelihood
    ------------
    Input:
    
    beta: can be a 1-by-N array or a N array
    game_matrix: records of games, N-by-N array
    ------------
    -l: negative loglikelihood, a number
    '''
    # beta could be a 1-by-N matrix or N array
    N = game_matrix.shape[0]
    # l stores the loglikelihood
    l = 0
    N_one = np.ones(N).reshape(N,1)
    Cu = np.triu(game_matrix)
    Cl = np.tril(game_matrix)
    b = beta.reshape(N,1)
    D = b @ N_one.T - N_one @ b.T
    W = np.log(1 + np.exp(D))
    l += N_one.T @ (Cu * D) @ N_one - N_one.T @ ((Cu + Cl.T) * np.triu(W)) @ N_one
    return -l[0,0]


def grad_nl(beta,game_matrix):
    '''
    calculate the gradient of the negative loglikelihood
    ------------
    Input:
    beta: can be a 1-by-N array or a N array
    game_matrix: records of games, N-by-N array
    ------------
    Output:
    -grad: gradient of negative loglikelihood, a T*N-by-1 array
    '''
    # beta could be a 1-by-N array or a N array
    N = game_matrix.shape[0]
    # g stores the gradient
    g = np.zeros(N).reshape(N,1)
    N_one = np.ones(N).reshape(N,1)
    
    C = game_matrix
    b = beta.reshape(N,1)
    W = np.exp(b @ N_one.T) + np.exp(N_one @ b.T)
    g = ((C / W) @ np.exp(b) - (C / W).T @ N_one * np.exp(b)).ravel()
    return -g.reshape(N,1)

def hess_nl(beta,game_matrix):
    
    N = game_matrix.shape[0]
    # H stores the Hessian
    H = np.zeros(N ** 2).reshape(N,N)
    N_one = np.ones(N).reshape(N,1)
    
    Cu = np.triu(game_matrix)
    Cl = np.tril(game_matrix)
    Tm = Cu + Cl.T + Cu.T + Cl
    b = beta.reshape(N,1)
    W = np.exp(b @ N_one.T) + np.exp(N_one @ b.T)
    H = Tm * np.exp(b @ N_one.T + N_one @ b.T) / W ** 2
    H += -np.diag(sum(H))
    
    return -H




def gd_bt(data,
              max_iter=1000, ths=1e-12,
              step_init=0.5, max_back=200, a=0.2, b=0.5,
              beta_init=None, verbose=False, out=sys.stdout):
    # initialize optimization
    N = data.shape[0]
    if beta_init is None:
        beta = np.zeros((N,1))
    else:
        beta = beta_init
    nll = neg_log_like(beta, data)

    # initialize record
    objective_wback = [nll]
    if verbose:
        out.write("initial objective value: %f\n"%objective_wback[-1])
        out.flush()

    # iteration
    for i in range(max_iter):
        # compute gradient
        gradient = grad_nl(beta, data).reshape([N,1])
        
        # backtracking line search
        s = step_init
        for j in range(max_back):
            beta_new = beta - s*gradient
            beta_diff = beta_new - beta
            
            nll_new = neg_log_like(beta_new, data)
            nll_back = (nll + np.sum(gradient * beta_diff) 
                        + np.sum(np.square(beta_diff)) / (2*s))
            
            if nll_new <= nll_back:
                break
            s *= b
        
        beta = beta_new
        nll = nll_new
        
        # record objective value
        objective_wback.append(neg_log_like(beta, data))
        
        if verbose:
            out.write("%d-th GD, objective value: %f\n"%(i+1, objective_wback[-1]))
            out.flush()
        if abs(objective_wback[-2] - objective_wback[-1]) < ths:
            if verbose:
                out.write("Converged!\n")
                out.flush()
            break
        elif i >= max_iter-1:
            if verbose:
                out.write("Not converged.\n")
                out.flush()

    return objective_wback, beta




########################## squared l2 penalty ############################


def objective_l2_sq(beta, game_matrix, l_penalty):
    '''
    compute the objective of the model (neg_log_like + l2_square)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + squared l2 penalty
    '''
    # reshape beta into TxN array
    N = game_matrix.shape[0]
    beta = beta.reshape([N,1])
    
    # compute l2 penalty
    l2_penalty = np.sum(np.square(beta))
    
    return neg_log_like(beta, game_matrix) + l_penalty * l2_penalty


def grad_l2_sq(beta, game_matrix, l):
    '''
    compute the gradient of the model (neg_log_like + l2_square)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + squared l2 penalty
    '''
    # reshape beta into TxN array
    N = game_matrix.shape[0]
    beta = beta.reshape([N,1])
    
    # compute l2 penalty
    l2_grad = grad_nl(beta, game_matrix)
    l2_grad += l * 2 * beta
    
    return  l2_grad


def hess_l2_sq(beta, game_matrix, l):
    '''
    compute the Hessian of the model (neg_log_like + l2_square)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + squared l2 penalty
    '''
    # reshape beta into TxN array
    N = game_matrix.shape[0]
    beta = beta.reshape([N,1])
    
    # compute l2 penalty
    l2_hess = hess_nl(beta, game_matrix)
    l2_hess += l * np.eye(N)
    
    return  l2_hess

def gd_l2_sq(data,l_penalty=1,
              max_iter=1000, ths=1e-12,
              step_init=0.5, max_back=200, a=0.2, b=0.5,
              beta_init=None, verbose=False, out=sys.stdout):
    # initialize optimization
    N = data.shape[0]
    if beta_init is None:
        beta = np.zeros((N,1))
    else:
        beta = beta_init
    nll = objective_l2_sq(beta, data, l_penalty)

    # initialize record
    objective_wback = [nll]
    if verbose:
        out.write("initial objective value: %f\n"%objective_wback[-1])
        out.flush()

    # iteration
    for i in range(max_iter):
        # compute gradient
        gradient = grad_l2_sq(beta, data, l_penalty).reshape([N,1])
        
        # backtracking line search
        s = step_init
        for j in range(max_back):
            beta_new = beta - s*gradient
            beta_diff = beta_new - beta
            
            nll_new = objective_l2_sq(beta_new, data, l_penalty)
            nll_back = (nll + np.sum(gradient * beta_diff) 
                        + np.sum(np.square(beta_diff)) / (2*s))
            
            if nll_new <= nll_back:
                break
            s *= b
        
        # proximal gradient update
        beta = beta_new
        nll = nll_new
        
        # record objective value
        objective_wback.append(objective_l2_sq(beta, data, l_penalty))
        
        if verbose:
            out.write("%d-th GD, objective value: %f\n"%(i+1, objective_wback[-1]))
            out.flush()
        if abs(objective_wback[-2] - objective_wback[-1]) < ths:
            if verbose:
                out.write("Converged!\n")
                out.flush()
            break
        elif i >= max_iter-1:
            if verbose:
                out.write("Not converged.\n")
                out.flush()

    return objective_wback, beta

def newton_l2_sq(data, l_penalty=1,
                 max_iter=1000, ths=1e-12,
                 step_init=1, max_back=200, a=0.01, b=0.3,
                 beta_init=None, verbose=False, out=sys.stdout):
    # initialize optimization
    N = data.shape[0]
    if beta_init is None:
        beta = np.zeros([N,1])
    else:
        beta = beta_init.reshape((N, 1))
    
    # initialize record
    objective_nt = [objective_l2_sq(beta, data, l_penalty)]
    if verbose:
        out.write("initial objective value: %f\n"%objective_nt[-1])
        out.flush()

    # iteration
    obj_old = objective_nt[0]
    for i in range(max_iter):
        # compute gradient
        gradient = grad_l2_sq(beta, data, l_penalty)
        hessian = hess_l2_sq(beta, data, l_penalty)
        # backtracking
        s = step_init
        beta_new = beta + 0 # make a copy
        
        v = -sc.linalg.solve(hessian, gradient)
        for j in range(max_back):
            beta_new = beta_new + s * v
            obj_new = objective_l2_sq(beta_new,data,l_penalty)
        
            if obj_new <= obj_old + b * s * gradient.T @ v:
                break
            s *= a
            
        beta = beta_new
        
        # objective value
        objective_nt.append(obj_new)
        obj_old = obj_new

        if verbose:
            out.write("%d-th Newton, objective value: %f\n"%(i+1, objective_nt[-1]))
            out.flush()
        if objective_nt[-2] - objective_nt[-1] < ths:
            if verbose:
                out.write("Converged!\n")
                out.flush()
            break
        elif i >= max_iter-1:
            if verbose:
                out.write("Not converged.\n")
                out.flush()

    beta = beta.reshape((1,N))
    beta = beta - sum(beta) / N   

    return objective_nt, beta