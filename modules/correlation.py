import math
import numpy
import scipy.io
import scipy.stats
import scipy.signal
import modules.io as io
import modules.misc_func as misc
from numba import njit
from enum import IntEnum

class SmoothingType(IntEnum):
    NONE       = 0
    GAUSSIAN   = 1
    MEXICAN    = 2
    MOVING_AVG = 3

class FilterType(IntEnum):
    NONE       = 0
    MEDIAN     = 1
    LOWPASS    = 2
    MOVING_AVG = 3
    WIENER     = 4

class PositionType(IntEnum):
    RING      = 0
    LINE      = 1
    LATTICE2D = 2
    RANDOM2D  = 3

@njit
def _largest_factors(N):
    for i in range(N // 2, 0, -1):
        if N % i == 0:
            return (N // i, i)
    return (N, 1)  # Numba prefers fixed types; avoid None


@njit
def calc_1d_positions_periodicBC(N):
    # position every site along a ring, since we are using periodic BC
    # r[k,:] -> (x,y) position of site k
    theta = numpy.linspace(0, 2.0 * numpy.pi, N + 1)[:N]
    r     = numpy.empty((N, 2))  # Pre-allocate the 2D position array
    for i in range(N):
        r[i, 0] = numpy.cos(theta[i])  # x-coordinate
        r[i, 1] = numpy.sin(theta[i])  # y-coordinate
    return r

@njit
def calc_1d_positions_freeBC(N):
    # position every site along the x-axis
    # r[k,:] -> (x,y) position of site k
    r = numpy.empty((N, 2))  # Pre-allocate the 2D position array
    for i in range(N):
        r[i, 0] = float(i)  # x-coordinate
        r[i, 1] = 0.0       # y-coordinate
    return r

@njit
def calc_2d_positions(N):
    r     = numpy.empty((N, 2))  # Pre-allocate the 2D position array
    Lx,Ly = _largest_factors(int(N))
    for i in range(Ly):
        for j in range(Lx):
            k = j + i * Lx
            r[k,0] = float(j)
            r[k,1] = float(i)
    return r

@njit
def calc_random_positions(N):
    Lx,Ly = _largest_factors(int(N))
    r     = numpy.empty((N, 2))  # Pre-allocate the 2D position array
    for i in range(N):
        r[i,0] = numpy.random.rand()*Lx
        r[i,1] = numpy.random.rand()*Ly
    return r

def calc_position(N, position:PositionType):
    if position == PositionType.RING:
        r = calc_1d_positions_periodicBC(N)
    elif position == PositionType.LINE:
        r = calc_1d_positions_freeBC(N)
    elif position == PositionType.LATTICE2D:
        r = calc_2d_positions(N)
    elif position == PositionType.RANDOM2D:
        r = calc_random_positions(N)
    else:
        raise ValueError('unknown position type')
    return r

def calc_correlation_function_MF(C,avg_same_distance=True, position:PositionType = 0):
    r = calc_position(C.shape[0], position)
    return calc_correlation_function_numba(C,r,avg_same_distance)

def calc_correlation_function_1d_periodicBC(C,avg_same_distance=True,position:PositionType = 1):
    r = calc_position(C.shape[0], position)
    return calc_correlation_function_numba(C,r,avg_same_distance)

def calc_correlation_function_1d_freeBC(C,avg_same_distance=True,position:PositionType = 1):
    r = calc_position(C.shape[0], position)
    return calc_correlation_function_numba(C,r,avg_same_distance)

@njit
def calc_correlation_function_numba(C,r,avg_same_distance):
    N  = C.shape[0]
    s  = [] #List.empty_list(numpy.float64) # we are defining C(s) as the correlation function
    Cf = [] #List.empty_list(numpy.float64) # we are defining C(s) as the correlation function
    for i in range(N):
        for j in range(i+1,N):
            s.append(numpy.linalg.norm(r[i,:]-r[j,:]))# distance between i,j
            Cf.append(C[i,j])
    if avg_same_distance:
        s_un,Cf_avg,Cf_std = calc_average_same_distance(numpy.array(s),numpy.array(Cf))
        k                  = numpy.argsort(s_un)
        return s_un[k],Cf_avg[k],Cf_std[k]
    else:
        k = numpy.argsort(numpy.array(s))
        return numpy.array(s)[k],numpy.array(Cf)[k],numpy.zeros_like(k,dtype=numpy.float64)

@njit
def calc_average_same_distance(s, Cf):
    # Round s to reduce floating-point precision issues
    s_rounded = numpy.round(s, decimals=8)  # You can adjust decimals as needed

    # Sort s and Cf together by s_rounded
    idx       = numpy.argsort(s_rounded)
    s_sorted  = s_rounded[idx]
    Cf_sorted = Cf[idx]

    # Initialize output lists
    s_un   = []#List.empty_list(numpy.float64)
    Cf_avg = []#List.empty_list(numpy.float64)
    Cf_std = []#List.empty_list(numpy.float64)

    # Average all Cf values that share the same s
    i = 0
    while i < len(s_sorted):
        sc = s_sorted[i]
        c  = Cf_sorted[i]
        c2 = Cf_sorted[i]*Cf_sorted[i]
        #sum_val   = Cf_sorted[i]
        count = 1
        i += 1
        while i < len(s_sorted) and s_sorted[i] == sc:
            c     += Cf_sorted[i]
            c2    += Cf_sorted[i]*Cf_sorted[i]
            count += 1
            i     += 1
        s_un.append(sc)
        Cf_avg.append(c / count)
        Cf_std.append(c2 / count - c*c/(count*count))
    return numpy.array(s_un), numpy.array(Cf_avg), numpy.array(Cf_std)

@njit
def my_exp(x): #definimos essa função para evitar o erro de overflow
    return math.exp(x) if x < 709.782712893384 else numpy.inf

@njit
def PoissonProcess_firingprob(r):
    return 1.0-my_exp(-r) # probability of firing is constant

@njit
def generate_Poisson_spikes(r, T, N):
    """
    Generates a matrix of Poisson spikes.
    Each column is an independent Poisson process with rate r.
    r -> Poisson rate [P=1-exp(-r) is the firing probability]
    T -> total number of time steps
    N -> number of independent processes (neurons)
    Returns a 2D numpy.ndarray of shape (T,N) where each element is either 0 or 1.
    """
    P = PoissonProcess_firingprob(r)
    X = numpy.zeros((T,N), dtype=numpy.float64)
    for k in range(N):
        X[numpy.random.rand(T)<P,k] = 1.0
    return X

def get_n_max_corrcoef(C,n=10,i=None,j=None):
    if type(C) is list:
        i,j = numpy.nonzero(numpy.tril(numpy.ones(C[0].shape)))
        ncf = misc.get_empty_list(len(C))
        ind = misc.get_empty_list(len(C))
        linind = misc.get_empty_list(len(C))
        for k,c in enumerate(C):
            ncf[k],ind[k],linind[k] = get_n_max_corrcoef(c,n,i=i,j=j)
    else:
        CC = C.copy()
        if (i is None) or (j is None):
            i,j = numpy.nonzero(numpy.tril(numpy.ones(C[0].shape)))
        CC[i,j] = -numpy.inf
        CC[numpy.isnan(CC)] = -numpy.inf
        CC = CC.flatten()
        k = numpy.argsort(CC)[-n:]
        ncf = CC[k]
        linind = k
        z = numpy.unravel_index(k,C.shape)
        k = numpy.lexsort((z[1],z[0]))
        ncf = ncf[k]
        linind = linind[k]
        ind = [ (z[0][i],z[1][i]) for i in k ]
        #ind = [ (i,j) for i,j in zip(z[0],z[1]) ]
    return ncf,ind,linind

def calc_correlation_distribution(C,nbins=100,smooth=False,ignoreZeroCorr=True):
    if smooth is True:
        smooth = 'average'
    if type(C) is list:
        n = len(C)
        P = misc.get_empty_list(n)
        bins = misc.get_empty_list(n)
        avg = misc.get_empty_list(n)
        std = misc.get_empty_list(n)
        for i,c in enumerate(C):
            P[i],bins[i],avg[i],std[i] = calc_correlation_distribution(c,nbins=nbins,smooth=smooth,ignoreZeroCorr=ignoreZeroCorr)
        return P, bins, avg, std
    else:
        x = C[numpy.eye(C.shape[0])!=1]
        if ignoreZeroCorr:
            x = x[x!=0]
        P,bins = numpy.histogram(x,bins=nbins,density=True)
        if smooth == 'savgol':
            P = scipy.signal.savgol_filter(P, 5, 2)
        elif smooth == 'average':
            P = moving_average(P,n=10)
        P = P / numpy.sum(P)
        avg = numpy.nanmean(x)
        std = numpy.nanstd(x)
        return P, bins[:-1], avg, std

def calc_null_correlation(S,ntrials=None,**corr_kwargs):
    """
    Computes a null (baseline) correlation matrix by averaging randomized versions of the original correlation matrix.

    This function generates a null model of the correlation structure in the spike time series `S` by repeatedly
    randomizing the off-diagonal elements of the original correlation matrix and averaging the results over multiple trials.

    Parameters:
    ----------
    S : ndarray
        A 2D array representing spike time series. Each row or column corresponds to a time point or neuron,
        depending on the `rowvar` flag.

    ntrials : int, optional
        Number of randomization trials to perform. If None, defaults to the number of time points in `S`.

    corr_kwargs : dict, optional
        Additional keyword arguments passed to `calc_correlation_matrices`.

    Returns:
    -------
    C : ndarray
        A null correlation matrix obtained by averaging `ntrials` randomized versions of the original matrix.

    Notes:
    -----
    - The original correlation matrix is computed using `calc_correlation_matrices`.
    - Randomization is applied only to the upper triangular off-diagonal elements.
    - The diagonal values are preserved or set to NaN depending on `nandiag`.
    """
    if ntrials is None:
        ntrials = S.shape[0]
    A,_ = calc_correlation_matrices(S,**corr_kwargs)
    C   = A.copy()
    i,j = numpy.nonzero(numpy.triu(numpy.ones(A.shape),k=1))
    for k in range(ntrials):
        C += rand_corr_matrix(A.copy(),i=i,j=j)
    return C / ntrials

def calc_dispersion_PCA(C):
    """
    Computes the eigenvalues, dispersion (standard deviation) of principal components,
    and eigenvectors from a covariance matrix using Principal Component Analysis (PCA).

    Parameters:
    -----------
    C : numpy.ndarray
        A square covariance matrix (n x n) representing the relationships (covariance matrix == physical correlation matrix) between variables.

    Returns:
    --------
    lambda_eig : numpy.ndarray
        Array of eigenvalues of the covariance matrix, representing the variance explained by each principal component.

    lambda_dispersion : numpy.ndarray
        Array of dispersions (standard deviations) of the principal components, calculated as the square root of the absolute eigenvalues.

    eigenvectors : list of numpy.ndarray
        List of eigenvectors corresponding to each principal component, each as a 1D array.

    V_matrix : numpy.ndarray
        Matrix whose columns are the eigenvectors of the covariance matrix.

    Notes:
    ------
    - The eigenvalues may be complex if the input matrix is not symmetric.
    - The function takes the absolute value of eigenvalues before computing the square root to ensure real-valued dispersions.
    """
    if type(C) is list:
        return misc.unpack_list_of_tuples([ _calc_dispersion_PCA_numba(c) for c in C ])
    return _calc_dispersion_PCA_numba(C)

#@njit
def _calc_dispersion_PCA_numba(C):
    #for i in range(C.shape[0]):
    #    for j in range(C.shape[1]):
    #        if numpy.isnan(C[i, j]):
    #            C[i, j] = 0.0
    C[numpy.isnan(C)]   = 0.0
    if numpy.all(C==C.T):
        lambda_eig,V_matrix = numpy.linalg.eigh(C)
    else:
        lambda_eig,V_matrix = numpy.linalg.eig(C)
    lambda_dispersion   = numpy.sqrt(numpy.abs(lambda_eig))
    eigenvectors        = [ V_matrix[:,m].flatten() for m in range(V_matrix.shape[1]) ]
    return lambda_eig,lambda_dispersion,eigenvectors,V_matrix

def calc_correlation_matrices(S, DeltaT=None, overlap_DeltaT=False,
    smooth : SmoothingType = False, smooth_args=None,
    binarize=True, spk_threshold=59.0,
    rowvar=True, nandiag=True,
    filterSpkFreq: FilterType = False, filter_args=None):
    """
    Compute one or more correlation (covariance) matrices from a spiketrain array `S`
    where each row corresponds to a neuron and each column corresponds to a time point.

    The function can segment the time series into intervals of length `DeltaT` (possibly
    overlapping) and compute a correlation matrix for each segment. Optional preprocessing
    steps include binarization, smoothing, and spike-frequency filtering.

    Parameters
    ----------
    S : ndarray of shape (N, T)
        Spiketrain matrix, where rows correspond to neurons and columns correspond to time points.
        Typically the output of `spike_times_to_spiketrain()`.
    DeltaT : int, optional
        Length of each time interval (in number of time points) used to compute an individual
        correlation matrix. If None, the entire time series is treated as one interval.
    overlap_DeltaT : bool, default=False
        If True, uses overlapping time intervals for correlation computation.
        Otherwise, uses adjacent non-overlapping windows.
    smooth : SmoothingType, optional
        Whether to apply smoothing to the spike data before correlation calculation.
        See `SmoothingType` for available options.
    smooth_args : dict, optional
        Additional arguments for the smoothing function, e.g.:
        ```
        dt=0.01, stddev=0.1, J=None, kernel_size=10
        ```
        - `stddev`: Standard deviation for Gaussian smoothing.
        - `J`: Extra parameter (used for e.g. 'mexican' smoothing).
    binarize : bool, default=True
        If True, converts the spike data to binary format using a threshold before smoothing/filtering.
    spk_threshold : float, default=59.0
        Threshold used for binarization; values above this are treated as spikes (1.0), others as 0.0.
    rowvar : bool, default=True
        Controls variable orientation for covariance calculation (as in `numpy.cov`):
        - If `True` (default), each row of `S` is a variable (neuron), and columns are time points.
        - If `False`, each column is a variable.
        Normally you should leave this as `False` when using `S` from `spike_times_to_spiketrain`.
    nandiag : bool, default=True
        If True, replaces the diagonal of each correlation matrix with NaN to ignore self-correlations.
    filterSpkFreq : FilterType, optional
        Whether to apply frequency-domain filtering to the spike series before computing correlations.
        See `FilterType` for available options.
    filter_args : dict, optional
        Additional arguments for spike-frequency filtering, e.g.:
        ```
        kernel_size=3, fs=10.0, cutoff=1.0, order=5, noise=None
        ```

    Returns
    -------
    C : ndarray or list of ndarrays
        - If only one interval is used, returns a single 2D array of shape (N, N),
          containing the neuron–neuron covariance (or correlation) matrix.
        - If multiple intervals are used, returns a list of such matrices, one per interval.
    tRange : tuple or list of tuples
        - If only one interval is used, returns a single tuple `(t_start, t_end)` giving
          the time range of that interval.
        - If multiple intervals are used, returns a list of tuples for each interval.

    Notes
    -----
    - The function uses covariance (`numpy.cov`) rather than correlation (`numpy.corrcoef`)
      for matrix computation.
    - NaN entries in the correlation matrices are replaced with 0.0, except along the diagonal
      if `nandiag=True`.
    - When both smoothing and filtering are applied, smoothing occurs first.
    - For overlapping windows (`overlap_DeltaT=True`), the function uses `_get_corr_timerange_overlap`
      to compute window boundaries; otherwise it uses `_get_corr_timerange_adjacent`.
    - Suitable for analyzing population-level synchrony or pairwise correlations between neurons.
    """
    tind = 1 if rowvar else 0
    if DeltaT is None:
        DeltaT = S.shape[tind]
    if binarize:
        S = get_binary_spike_series(S,spk_threshold=spk_threshold)
    if smooth:
        S = smooth_spikes(S,smooth=smooth,rowvar=rowvar,**misc._get_kwargs(smooth_args))
    if filterSpkFreq:
        S = filter_spikes(S,rowvar=rowvar,**misc._get_kwargs(filter_args))
    if overlap_DeltaT:
        nT             = S.shape[tind] - 1
        get_time_range = _get_corr_timerange_overlap
    else:
        nT             = int(numpy.ceil(float(S.shape[tind]) / float(DeltaT)))
        get_time_range = _get_corr_timerange_adjacent
    
    C                                = misc.get_empty_list(nT)
    tRange                           = misc.get_empty_list(nT)
    if rowvar:
        get_time_range_from_spike_matrix = lambda S,t1,t2: S[:,t1:t2]
    else:
        get_time_range_from_spike_matrix = lambda S,t1,t2: S[t1:t2,:]
    for n in range(nT):
        t1,t2 = get_time_range(n,DeltaT)
        t2    = min(S.shape[tind],t2)
        if (t2-t1) > 2:
            tRange[n] = (t1,t2)
            C[n] = numpy.cov(get_time_range_from_spike_matrix(S,t1,t2),rowvar=rowvar) #numpy.corrcoef(S[t1:t2,:],rowvar=rowvar)
            C[n][numpy.isnan(C[n])] = 0.0
            #if numpy.count_nonzero(numpy.isnan(C[i])) > 0:
            #print('index == %d -> [%d;%d]     ---- number of NaN: %d' % (i,t1,t2,numpy.count_nonzero(numpy.isnan(C[i]))))
            if nandiag:
                numpy.fill_diagonal(C[n],numpy.nan)

    if nT == 1: # squeeze output
        C      = C[0]
        tRange = tRange[0]
    else: # remove None values
        C      = [c for c in C      if c is not None]
        tRange = [t for t in tRange if t is not None] 
    return C, tRange

def filter_spikes(S,filter_type:FilterType=None,kernel_size=3,fs=10.0,cutoff=1.0,order=5,noise=None,rowvar=False):
    if (filter_type == FilterType.NONE) or (filter_type is False):
        return S
    #if rowvar:
    #    S        = S.T
    #    return_S = lambda S: S.T
    axis = 1 if rowvar else 0
    if filter_type == FilterType.MEDIAN:
        return filter_spk_freq_median(S, kernel_size,rowvar=rowvar)
    elif filter_type == FilterType.LOWPASS:
        return filter_butter_lowpass(S, cutoff, fs, order,rowvar=rowvar)
    elif filter_type == FilterType.MOVING_AVG:
        return numpy.apply_along_axis(moving_average, axis, S, n=kernel_size)
    elif filter_type == FilterType.WIENER:
        return numpy.apply_along_axis(scipy.signal.wiener, axis, S, mysize=kernel_size, noise=noise)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}. Supported types are given by FilterType enum.")

def _filter_create_butter_lowpass(cutoff, fs, order=5):
    return scipy.signal.butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')

def filter_butter_lowpass(S, cutoff, fs, order=5,rowvar=False):
    #if rowvar:
    #    S = S.T
    #b, a = _filter_create_butter_lowpass(cutoff, fs, order=order)
    #n    = S.shape[1]
    #for i in range(n):
    #    S[:,i] = scipy.signal.filtfilt(b, a, S[:,i])
    #return S.T if rowvar else S
    sos  = _filter_create_butter_lowpass(cutoff, fs, order=order)
    axis = 1 if rowvar else 0
    return numpy.apply_along_axis(lambda x: scipy.signal.sosfiltfilt(sos, x), axis, S)

def filter_spk_freq_median(S,kernel_size=3,rowvar=False):
    """ to be implemented: remove background spikes using median filter """
    #if rowvar:
    #    S = S.T
    #n = S.shape[1]
    #for i in range(n):
    #    S[:,i] = scipy.signal.medfilt(S[:,i],kernel_size)
    #return S.T if rowvar else S
    axis = 1 if rowvar else 0
    return numpy.apply_along_axis(scipy.signal.medfilt, axis, S, kernel_size)

def filter_corrmatrix_null_avg(C,null_avg):
    if type(C) is list:
        for n in range(len(C)):
            C[n] = filter_corrmatrix_null_avg(C[n],null_avg)
    else:
        C[numpy.nonzero(C < null_avg)] = 0.0
    return C

def _get_corr_timerange_overlap(n,DeltaT):
    return n, n+DeltaT

def _get_corr_timerange_adjacent(n,DeltaT):
    return n*DeltaT, (n+1)*DeltaT

def rand_corr_matrix(A, i=None, j=None):
    """
    Randomizes the off-diagonal upper triangular elements of a symmetric correlation matrix.

    This function generates a randomized version of a symmetric matrix `A` by shuffling its upper triangular
    off-diagonal elements. The diagonal is preserved, and symmetry is maintained by mirroring the randomized
    upper triangle to the lower triangle.

    Parameters:
    ----------
    A : ndarray
        A square symmetric matrix (typically a correlation or covariance matrix).

    i : ndarray, optional
        Row indices of the upper triangular off-diagonal elements to be randomized.
        If None, these indices are automatically computed.

    j : ndarray, optional
        Column indices corresponding to `i`. If None, these are automatically computed.

    Returns:
    -------
    A_rand : ndarray
        A symmetric matrix with the same diagonal as `A`, but with randomized off-diagonal elements.

    Notes:
    -----
    - Only the upper triangular part (excluding the diagonal) is randomized.
    - The resulting matrix is symmetric: `A_rand[i, j] == A_rand[j, i]`.
    - Useful for generating null models or testing statistical significance of correlation structures.
    """
    if i is None or j is None:
        i,j = numpy.nonzero(numpy.triu(numpy.ones(A.shape),k=1))
    x = A[i,j] # gets all the upper triangular elements in A
    x = x[numpy.random.permutation(len(x))]
    A[i,j] = x
    return numpy.triu(A,k=1) + numpy.tril(numpy.transpose(A))

def get_binary_spike_series(S,spk_threshold=0.0):
    return numpy.asarray(S>spk_threshold,dtype=float)

def smooth_spikes(S, smooth:SmoothingType=None, dt=0.01, stddev=0.1, J=None, kernel_size=10, rowvar=False):
    """
    Applies temporal smoothing to binary spike time series using convolution.

    This function smooths each column of the input matrix `S`, which represents binary spike trains (0s and 1s),
    by convolving them with a specified smoothing kernel. If no kernel is provided, a Gaussian kernel is used by default.

    Parameters:
    ----------
    S : ndarray
        A 2D array where each column represents a binary spike time series for a neuron.
    smooth : SmoothingType, optional
        The smoothing kernel to use. See SmoothingType for options.
    dt : float, optional
        Time resolution of the spike data (used when generating kernels). Default is 0.01.
    stddev : float, optional
        Standard deviation of the Gaussian or Mexican hat kernel. Controls the smoothing scale.
    J : float or None, optional
        Additional parameter used when generating the Mexican hat kernel.
    kernel_size : int, optional
        kernel size used for moving average if 'movingavg' is specified as 'smoothFunc'. Default is 10.

    Returns:
    -------
    S_smooth : ndarray
        A 2D array of the same shape as `S`, where each column has been smoothed via convolution.

    Notes:
    -----
    - Convolution is performed with `mode='same'`, preserving the original length of each spike train.
    - This function is useful for estimating firing rates or preparing spike data for correlation analysis.
    """
    #if rowvar:
    #    S = S.T
    #N = S.shape[1]
    if (smooth == SmoothingType.NONE) or (smooth is False):
        return S
    if smooth == SmoothingType.MOVING_AVG:
        kernel = numpy.ones(kernel_size,dtype=float) / float(kernel_size)
    elif smooth == SmoothingType.GAUSSIAN:
        kernel = get_gaussian_kernel(stddev,dt=dt)
    elif smooth == SmoothingType.MEXICAN:
        kernel = get_mexican_hat_kernel(stddev, J=J, dt=dt)
    else:
        raise ValueError(f"Unknown smoothing type: {smooth}. Supported types are given by SmoothingType enum.")
    #for i in range(N):
    #    S[:,i] = numpy.convolve(S[:,i],kernel,mode='same')
    #return S.T if rowvar else S
    axis = 1 if rowvar else 0
    return numpy.apply_along_axis(numpy.convolve,axis,S,kernel,'same')

def moving_average(x, n=10):
    return numpy.convolve(x,numpy.ones(n)/n,mode='same')

def get_mexican_hat_kernel(sigma1, J=None, dt=None):
    if J is None:
        J=4.0*sigma1
    if dt is None:
        dt = 0.001
    sigma2 = numpy.sqrt(numpy.power(sigma1, 2.) + numpy.power(J, 2.))
    if sigma2 < sigma1:
        sigma1, sigma2 = sigma2, sigma1
    k1 = get_gaussian_kernel(sigma1,dt=dt)
    k2 = get_gaussian_kernel(sigma2,dt=dt)
    n2 = k2.shape[0]
    n1 = k1.shape[0]
    m = int( numpy.floor((n2-n1)/2.0) )
    n = int( numpy.ceil((n2-n1)/2.0) )
    return numpy.pad(k1,(m,n))-k2

def get_gaussian_kernel(sigma,dt=0.01):
    t = numpy.arange(-3*sigma,3*sigma,dt)
    G = scipy.stats.norm.pdf(t,scale=sigma) * dt
    return G / numpy.sum(G)

@njit
def _append(arr, val):
    """Numba-safe version of numpy.append for 1D arrays."""
    n       = arr.size
    out     = numpy.empty(n + 1, arr.dtype)
    out[:n] = arr
    out[n]  = val
    return out

@njit
def _insert(arr, idx, val):
    """Numba-safe version of numpy.insert for 1D arrays."""
    n           = arr.size
    out         = numpy.empty(n + 1, arr.dtype)
    out[:idx]   = arr[:idx] # copy up to insertion point
    out[idx]    = val # insert value
    out[idx+1:] = arr[idx:] # copy rest
    return out

@njit
def _convert_activation_deactivation_to_state(S):
    """
    converts the events in S[n,t] into a state matrix M[n,t]
    where S[n,t] = +1 or -1 (activation or deactivation event)
    and M[n,t] = 1 or 0 (active or inactive state)
    S -> matrix of activation (+1) and deactivation (-1) events
    returns
        M -> state matrix M[n,t] = 1=active or 0=inactive
    """
    a = numpy.where(S > 0)[0] # activation
    b = numpy.where(S < 0)[0] # deactivation
    has_elem_a = a.size>0
    has_elem_b = b.size>0
    has_elem   = has_elem_a and has_elem_b
    if (has_elem_b and not has_elem_a) or (has_elem and (b[0] < a[0])): # if the first event is a deactivation
        a = _insert(a,0,0) # we assume the site was active at t=0
    if (has_elem_a and not has_elem_b) or (has_elem and (a[-1] > b[-1])): # if the last event is an activation
        b = _append(b,S.size-1) # we assume the site was active until the end
    for t1,t2 in zip(a,b):
        S[t1:(t2+1)] = 1 # extending the activation until the next deactivation
    return S

@njit
def _spike_times_to_spiketrain_numba(t, n, X, T, N, use_X, convert_act_deact_events_to_site_state=False, use_cumul_sum=True):
    """
    Convert spike time and neuron index arrays into a spiketrain matrix using Numba for performance.

    Parameters:
    - t (ndarray of int32): Array of spike times.
    - n (ndarray of int32): Array of neuron indices corresponding to each spike time.
    - X (ndarray of float64): Array of spike magnitudes or weights.
    - T (int): Maximum time index (defines number of time steps).
    - N (int): Number of neurons.
    - use_X (bool): If True, use values from X; otherwise, use 1.0 for each spike.

    Returns:
    - S (ndarray of shape (N, T+1)): A 2D spiketrain matrix where each entry S[n, t] represents
      the spike value (either from X or 1.0) for neuron `n` at time `t`.
    """
    S = numpy.zeros((N,T+1), dtype=numpy.int32)
    for i in range(t.size):
        time   = t[i]
        neuron = n[i]
        S[neuron,time] += X[i] if use_X else 1
    if convert_act_deact_events_to_site_state:
        for i in range(N):
            if use_cumul_sum:
                S[i,:] = numpy.cumsum(S[i,:])
            else:
                S[i,:] = _convert_activation_deactivation_to_state(S[i,:])
    return S

def _is_integer(num):
    return isinstance(num, int) or (isinstance(num, float) and num.is_integer())

def spike_times_to_spiketrain(t, n, X=None, T=None, N=None, 
                              convert_act_deact_events_to_site_state=False, 
                              use_cumul_sum=True, 
                              compress_time=True):
    """
    Generate a spiketrain matrix from spike times and neuron indices, optionally using spike magnitudes.
    
    The function converts lists of spike times (`t`) and corresponding neuron indices (`n`) into a
    2D spiketrain matrix `S` of shape `(N, T+1)`, where each row represents a neuron and each column
    represents a time step. Entries `S[n, t]` contain the spike value (default 1.0 unless `X` is provided).

    Parameters
    ----------
    t : array-like of int
        Spike times (time indices). Non-integer values are rounded and converted to int32.
        If `compress_time=True`, these times are remapped into a sequential integer range
        preserving the repetition pattern (e.g., `[0,0,0,1000,1000,100000] → [0,0,0,1,1,2]`).
        This avoids allocating extremely sparse arrays in memory.
    n : array-like of int
        Neuron indices corresponding to each spike in `t`. Must have the same length as `t`.
    X : array-like, optional
        Spike magnitudes or values. If provided, must have the same shape as `t`.
        If omitted, all spikes are treated as having value `1.0`.
    T : int, optional
        Maximum time index (i.e., number of time steps - 1). If None, inferred from `max(t)`.
        Ignored if `compress_time=True` since compressed time is reindexed.
    N : int, optional
        Total number of neurons. If None, inferred from `max(n)`.
    convert_act_deact_events_to_site_state : bool, default=False
        If True, converts activation/deactivation spike events into binary site states
        using cumulative summation logic (useful for on/off event representations).
    use_cumul_sum : bool, default=True
        If True, cumulative summation is used when integrating spike events over time.
        Typically used to maintain activation state when spikes represent transitions.
    compress_time : bool, default=True
        If True, remaps sparse or large spike time values into sequential integer indices
        while preserving their relative order and repetition structure.
        This greatly reduces memory usage for sparse `t`.

    Returns
    -------
    S : ndarray of shape (N, T+1)
        A 2D spiketrain matrix. Each entry `S[n, t]` contains the spike value (either from `X`
        or 1.0 if not provided) for neuron `n` at time index `t`.

    Notes
    -----
    - When `compress_time=True`, the returned `S` uses a compressed time axis. You can retrieve
      the mapping between compressed and original times using `numpy.unique(t_original, return_inverse=True)`
      before calling this function if you need to track the correspondence.
    - Non-integer time values in `t` are automatically rounded with a warning.
    - This function handles input validation and preprocessing before delegating to a 
      Numba-accelerated implementation (`_spike_times_to_spiketrain_numba`) for performance.
    """
    if not all(_is_integer(tt) for tt in t):
        print(' ::: WARNING ::: Converting spike times to integers... If this is not desired, please convert them before calling, e.g., using t/dt')
    t = numpy.asarray(t, dtype=numpy.int32)
    T = int(T) if misc.exists(T) else int(numpy.max(t))
    if compress_time:
        _,t = numpy.unique(t, return_inverse=True)
        T   = numpy.max(t)
    n = numpy.asarray(n, dtype=numpy.int32)
    N = int(N) if misc.exists(N) else int(numpy.max(n))
    if misc.exists(X):
        X     = numpy.asarray(X, dtype=numpy.int32)
        use_X = True
        assert X.shape == t.shape, 'X must match shape of t'
    else:
        X     = numpy.ones_like(t, dtype=numpy.float64)
        use_X = False
    return _spike_times_to_spiketrain_numba(t, n, X, T, N, use_X, convert_act_deact_events_to_site_state,use_cumul_sum)

def calc_firing_rate_from_spiketrain(S,is_sequential_update=True):
    """
    converts spike trains (or event matrix) into a firing rate
    S[n,t]               -> spike train (or event matrix in the case of sequential updates)
                            of site n at time t
    is_sequential_update -> if True, assumes S[n,t] contains both activations (+1) and deactivations (-1);
                            if False, assumes S[n,t] contains spike events only
    returns
        rho[t] -> firing rate at time t
    """
    sum_events = S.sum(axis=0)
    if is_sequential_update:
        sum_events = numpy.cumsum(sum_events)
    return sum_events/S.shape[0]

def calc_firing_rate_from_spike_times(time,N,X_time,X_values=None,is_sequential_update=True):
    """
    converts spikes (or events) times into a firing rate
    time                 -> vector of time points
    N                    -> number of sites (or neurons)
    X_time               -> vector of time points when events occur (either activation of deactivation if sequential updates)
    X_values             -> vector of values of events
                                1 for activation, -1 for deactivation (if sequential updates is True);
                                only 1 for all spike events (if not sequential updates)
    is_sequential_update -> if True, assumes X_values contains both activations (+1) and deactivations (-1);
                            if False, assumes X_values contains spike events only at each time point
    returns
        rho[t] -> firing rate at time t
    """
    if not misc.exists(X_values):
        print(' ::: WARNING ::: Assuming X_values=1 and not sequential updates...')
        X_values             = numpy.ones_like(X_time,dtype=int)
        is_sequential_update = False
    S0 = numpy.nonzero(X_time==time[0])[0].size
    S  = numpy.array([ (X[0] if ((X:=X_values[X_time==t]).size) else 0) for t in time[1:] ])
    if is_sequential_update:
        S = (numpy.cumsum(S)+S0)
    else:
        S = numpy.insert(S,0,S0)
    return S/N

#def spike_times_to_spiketrain(t,n,X=None,T=None,N=None):
#    t = (t if _is_numpy_array(t) else numpy.asarray(t)).astype(int)
#    n = (n if _is_numpy_array(n) else numpy.asarray(n)).astype(int)
#    T = int(T if misc.exists(T) else numpy.max(t))
#    N = int(N if misc.exists(N) else numpy.max(n))
#    if misc.exists(X):
#        X = X if _is_numpy_array(X) else numpy.asarray(X)
#        assert X.shape == t.shape, 'The input X must be the state of each node n at time t, so it must match the type and shape of t.'
#    _type = X.dtype if misc.exists(X) else float
#    S     = numpy.zeros((T+1,N),dtype=_type)
#    for i in range(N):
#        ind         = numpy.nonzero(n == i)[0]
#        S[t[ind],i] = X[ind] if misc.exists(X) else 1.0
#    return S

def membpotential_to_spiketrain(V_data,t=None,spk_threshold=59.0):
    # converts each column of data into a numpy binary array of spike trains
    # t is the time vector
    (T,N) = V_data.shape
    spktrains = misc.get_empty_list(N)
    if t is None:
        t = numpy.arange(T)
    #dt = numpy.mean(numpy.squeeze(numpy.diff(t)))
    for j in range(N):
        spktrains[j] = numpy.zeros(V_data[:,j].size, dtype=float)
        spktrains[j][V_data[:,j]>spk_threshold] = 1.0  #neo.SpikeTrain(t[numpy.nonzero(data[:,j] > spk_threshold)], units='ms', t_start=t[0], t_stop=t[-1])
    return spktrains

def membpotential_to_spike_times(V_data,t=None,spk_threshold=59.0):
    # converts each column of data into an array of spike times
    # t is the time vector
    (T,N) = V_data.shape
    spktimes = misc.get_empty_list(N)
    if t is None:
        t = numpy.arange(T)
    for j in range(N):
        spktimes[j] = t[numpy.nonzero(V_data[:,j] > spk_threshold)]
    return spktimes

def save_correlation_data(fname,C_data,lmbda_data,lmbda_dispersion_data,eigenvectors_data,V_matrix_data,s_data,Cf_data,Cf_std_data,d_info):
    return scipy.io.savemat(fname, dict(
            C_data                = io.list_of_arr_to_arr_of_obj(C_data)                 , 
            lmbda_data            = io.list_of_arr_to_arr_of_obj(lmbda_data)             , 
            lmbda_dispersion_data = io.list_of_arr_to_arr_of_obj(lmbda_dispersion_data)  , 
            eigenvectors_data     = io.list_of_arr_to_arr_of_obj(eigenvectors_data)      , 
            V_matrix_data         = io.list_of_arr_to_arr_of_obj(V_matrix_data)          , 
            s_data                = io.list_of_arr_to_arr_of_obj(s_data)                 , 
            Cf_data               = io.list_of_arr_to_arr_of_obj(Cf_data)                , 
            Cf_std_data           = io.list_of_arr_to_arr_of_obj(Cf_std_data)            , 
            d_info                = io.structtype_to_recarray(d_info)                    ) , appendmat=True, do_compression=True)

def load_correlation_data(fname,split_variables=True):
    corr_data                 = scipy.io.loadmat(fname,squeeze_me=True)
    if split_variables:
        C_data                = corr_data['C_data']
        lmbda_data            = corr_data['lmbda_data']
        lmbda_dispersion_data = corr_data['lmbda_dispersion_data']
        eigenvectors_data     = corr_data['eigenvectors_data']
        V_matrix_data         = corr_data['V_matrix_data']
        s_data                = corr_data['s_data']
        Cf_data               = corr_data['Cf_data']
        Cf_std_data           = corr_data['Cf_std_data']
        d_info                = io.recarray_to_structtype(corr_data['d_info'])
        return C_data,lmbda_data,lmbda_dispersion_data,eigenvectors_data,V_matrix_data,s_data,Cf_data,Cf_std_data,d_info
    else:
        corr_data['d_info']   = io.recarray_to_structtype(corr_data['d_info'])
        return corr_data
