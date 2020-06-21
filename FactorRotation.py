import numpy as np
import matplotlib.pyplot as plt

class Rotation(object):
    """
    This class includes various rotation methods. 
    orthmax, quartimax and varimax can be used for estimating orthogonal rotation matrices.
    oblimin, quartimin, biquartimin, covarimin and geomin can be used for estimating oblique rotation matrices.
    all oblique methods return Rotated loadings, Rotation matrix and Factor correlation matrix,
    while all orthogonal methods return Rotated loadings, Rotation matrix.
    This program is based on Jennrich(2002).
    
    parameters:
    * maxiter: iteration limit
    * nstarts: number of starting point
    * tol: it defines strictness of convergence criteria
    * A: factor loadings which will be rotated to have simple structure
    * gam: it determines rotational method.
    * n0: number of zeros to be prespecified for simplimax rotation
    
    returns:
    * A@T, A@np.linalg.inv(T.T) : rotated loadings
    * T, T_hat: estimated rotational matrix
    * T_hat.T@T_hat: (oblique only)factor correlation matrix
    
    ### Example ###
    Lam = np.array([[0.81,0.,0.],
                    [0.71,0.,0.],
                    [0.61,0.,0.],
                    [0.,0.72,0.],
                    [0.,-0.5,0.],
                    [0.,0.,-0.9],
                    [0.,0.,-0.72]])
                
    np.random.seed(100)
    T1 = np.linalg.svd(np.random.uniform(size=9).reshape(3,3))[0]
    Lam_contaminated = Lam@T1 # rotate Lam randomly
    Lam_contaminated
    
    rot = Rotation() # generate an instance
    rot.varimax(Lam_contaminated) # varimax rotation
    rot.geomin(Lam_contaminated) # geomin rotation
    rot.simplimax(Lam_contaminated, n0 = 14) # simplimax rotation
    
    """
    
    def __init__(self, maxiter = 1000, nstarts = 10, tol = 1e-4):
        self.maxiter = 1000
        self.nstarts = nstarts
        self.tol = tol
        print("         Orthogonal rotation: orthomax, quartimax, varimaxm, procrustes\n \
        Oblique rotation: oblimin, quartimin, biquartimin, covarimin, geomin, simplimax\n \
        are available.")
        
    # Orthogonal rotations
    def _orth_proj(self, X):
        svd_X = np.linalg.svd(X)
        return svd_X[0]@svd_X[2]
    
    def _f_orthmax(self, A, T, gam):
        L = A@T
        L2 = L*L
        L2_bar = np.average(L2, axis = 0)
        return np.trace(L2.T@(L2-gam*L2_bar))/4
    
    def orthomax(self, A, gam=0.5):
        p, nfac = A.shape
        T_ini = np.random.uniform(size = nfac**2).reshape(nfac, nfac)
        T = self._orth_proj(T_ini)
        h0 = self._f_orthmax(A, T, gam)
        for _ in range(self.maxiter):
            alpha = 1
            L = A@T
            L2 = L*L
            G_q = L*(L2 - gam*np.average(L2, axis = 0))
            G = A.T@G_q
            T_new = self._orth_proj(G)
            h_alpha = self._f_orthmax(A,T_new,gam)
            T = T_new
        return A@T, T
    
    def quartimax(self, A):
        return self.orthomax(A, gam=0)
    
    def varimax(self, A):
        return self.orthomax(A, gam=1)
    
    def proc_orth(self, A, B):
        BA = B.T@A
        svd_AB = np.linalg.svd(BA)
        T = svd_AB[2].T@svd_AB[0].T
        return A@T, T
        
        
    
    # Oblique rotations
    def _oblique_proj(self, TT):
        return TT@np.diag(1/np.sqrt(np.diag(TT.T@TT)))
    
    def _f_oblimin(self, A, T, gam):
        p, nfac = A.shape
        L = A@np.linalg.inv(T.T)
        L2 = L*L
        C = np.ones((p,p))/p
        N = np.ones((nfac, nfac)) - np.eye(nfac)
        return np.trace(L2.T@((np.eye(p) - gam*C)@L2@N))/4
    
    def oblimin(self, A, gam = 0.3, trace = False):
        p, nfac = A.shape
        f_hat = np.Inf
        C = np.ones((p,p))/p
        N = np.ones((nfac, nfac)) - np.eye(nfac)
        for strt in range(self.nstarts):
            if trace:
                print("----epoc: %d ----" %(strt))
            T_ini = np.random.uniform(size = nfac**2).reshape(nfac, nfac)
            T = self._oblique_proj(T_ini)
            h0 = self._f_oblimin(A, T, gam)
            for _ in range(self.maxiter):
                alpha = 1
                L = A@np.linalg.inv(T.T)
                L2 = L*L
                Gq = L*((np.eye(p) - gam*C)@L2@N)
                G = -(L.T@Gq@np.linalg.inv(T)).T
                T_new = self._oblique_proj(T - alpha*G)
                h_alpha = self._f_oblimin(A, T_new, gam)
                while h_alpha >= h0:
                    alpha /= 2
                    T_new = self._oblique_proj(T - alpha*G)
                    h_alpha = self._f_oblimin(A, T_new, gam)
                if trace:
                    print(h_alpha)
                """
                # Stopping criteria suggested in Jennrich(2002) is as follows.
                # this comes from tha fact that T is a stationary point of f(rotation criterion)
                # defined on M(manifold) if and only if G = Tdiag(T'G). But it doesn't work well in my program,
                # so I replaced this to whether rotation criterion is converged or not
                #s = _sabun(G, T_new)
                #if s <= tol:
                    #break
                """
                if h0 - h_alpha <= self.tol:
                    if h_alpha <= f_hat:
                        f_hat = h_alpha
                        T_hat = T_new
                    break
                if _+1 == maxiter:
                    print("reached iteration limit!!!")
                    if h_alpha <= f_hat:
                        T_hat = T_new
                        f_hat = h_alpha
                    break
                T = T_new
                h0 = h_alpha
        return A@np.linalg.inv(T_hat.T), T_hat, T_hat.T@T_hat
    
    def quartimin(self, A, trace = False):
        return self.oblimin(A, gam = 0, trace = trace)
    
    def biquartimin(self, A, trace = False):
        return self.oblimin(A, gam = -1/2, trace = trace) # gam is 1/2 for quartimin in Jennrich(2002), it may be -1/2 
    
    def covarimin(self, A, trace = False):
        return self.oblimin(A, gam = -1, trace = trace) # gam is 1 for covarimin in Jennrich(2002), if may be -1
    
    
    def _f_geomin(self, A, T, eps = 0.01):
        p, nfac = A.shape
        L = A@np.linalg.inv(T.T)
        u = np.ones(p)
        v = np.ones(nfac)
        L2 = L*L
        return u@np.exp((1/nfac)*np.log(L2 + eps)@v)
    
    def geomin(self, A, trace = False, eps = 0.01): # from Jennrich(2004)
        p, nfac = A.shape
        f_hat = np.Inf
        u = np.ones(p)
        v = np.ones(nfac)
        for strt in range(self.nstarts):
            if trace:
                print("----epoc: %d ----" %(strt))
            T_ini = np.random.uniform(size = nfac**2).reshape(nfac, nfac)
            T = self._oblique_proj(T_ini)
            h0 = self._f_geomin(A, T)
            for _ in range(self.maxiter):
                alpha = 1
                L = A@np.linalg.inv(T.T)
                L2 = L*L
                Gq = (2/nfac)*1/(L2 + eps)*L*np.outer(np.exp((1/nfac)*np.log(L2+eps)@v), v)
                G = -(L.T@Gq@np.linalg.inv(T)).T
                T_new = self._oblique_proj(T - alpha*G)
                h_alpha = self._f_geomin(A, T_new)
                while h_alpha >= h0:
                    alpha /= 2
                    T_new = self._oblique_proj(T - alpha*G)
                    h_alpha = self._f_geomin(A, T_new)
                if trace:
                    print(h_alpha)
            
                if h0 - h_alpha <= self.tol:
                    if h_alpha <= f_hat:
                        f_hat = h_alpha
                        T_hat = T_new
                    break
                if _+1 == maxiter:
                    print("reached iteration limit!!!")
                    if h_alpha <= f_hat:
                        T_hat = T_new
                        f_hat = h_alpha
                    break
                T = T_new
                h0 = h_alpha
        return A@np.linalg.inv(T_hat.T), T_hat, T_hat.T@T_hat
    
    def _f_simplimax(self, A, T, W):
        L = A@np.linalg.inv(T.T)
        return np.trace((W*L).T@(W*L))
    
    def simplimax(self, A, n0 = 0, trace = False):
        p, nfac = A.shape
        f_hat = np.Inf
        u = np.ones(p)
        v = np.ones(nfac)
        for strt in range(nstarts):
            if trace:
                print("----epoc: %d ----" %(strt))
            T_ini = np.random.uniform(size = nfac**2).reshape(nfac, nfac)
            T = self._oblique_proj(T_ini)
            h0 = np.Inf
            for _ in range(maxiter):
                alpha = 1
                L = A@np.linalg.inv(T.T)
                vecL = np.sort(np.abs(L).reshape(p*nfac))
                W = np.zeros((p, nfac))
                W[abs(L) <= vecL[n0-1]] = 1
                Gq = 2*W*L
                G = -(L.T@Gq@np.linalg.inv(T)).T
                T_new = self._oblique_proj(T - alpha*G)
                h_alpha = self._f_simplimax(A, T_new, W)
                inner_cnt = 0
                while h_alpha >= h0 and inner_cnt < 10:
                    inner_cnt += 1
                    alpha /= 2
                    T_new = self._oblique_proj(T - alpha*G)
                    h_alpha = self._f_simplimax(A, T_new, W)
                if trace:
                    print(h_alpha)
                """
                # Stopping criteria suggested in Jennrich(2002) is as follows.
                #
                #s = _sabun(G, T_new)
                #if s <= tol:
                #    #break
                #    
                # this comes from tha fact that T is a stationary point of f(rotation criterion)
                # defined on M(manifold) if and only if G = Tdiag(T'G). But it doesn't work well in my program,
                # so I replaced this to whether rotation criterion is converged or not
                """
                if h0 - h_alpha <= tol:
                    if h_alpha <= f_hat:
                        f_hat = h_alpha
                        T_hat = T_new
                    break
                if _+1 == maxiter:
                    print("reached iteration limit!!!")
                    if h_alpha <= f_hat:
                        T_hat = T_new
                        f_hat = h_alpha
                    break
                T = T_new
                h0 = h_alpha
        return A@np.linalg.inv(T_hat.T), T_hat, T_hat.T@T_hat
        
    """
    
    T_new, h_alpha = update_T()
    
    def update_T(A, G_q, T, Qrot, ,h0):
        alpha = 1
        G = -(L.T@Gq@np.linalg.inv(T)).T
        T_new = self._oblique_proj(T - alpha*G)
        h_alpha = self.Qrot(A, T_new)
        while h_alpha >= h0:
            alpha /= 2
            T_new = self._oblique_proj(T - alpha*G)
            h_alpha = self.Qrot(A, T_new)
        return T_new, h_alpha
        
    def oblique_rot(A, gam = 0.3, n0 = 0, eps = 0.01, trace = False):
        p, nfac = A.shape
        f_hat = np.Inf
        u = np.ones(p)
        v = np.ones(nfac)
    """