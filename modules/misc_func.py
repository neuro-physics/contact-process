import re
import copy
import numpy
import warnings
import scipy.stats
import collections
from numba import njit, types

def get_empty_list(N):
    return [ None for _ in range(N)]

def _make_scalar(x):
    return x[0] if len(x) == 1 else x

def set_default_kwargs(kwargs_dict,**default_args):
    """
    kwargs_dict is the '**kwargs' argument of any function
    this function checks whether any argument in kwargs_dict has a default value given in default_args...
    if yes, and the corresponding default_args key is not in kwargs_dict, then includes it there

    this is useful to avoid duplicate key errors
    """
    kwargs_dict = copy.deepcopy(kwargs_dict)
    for k,v in default_args.items():
        if not (k in kwargs_dict):
            kwargs_dict[k] = v
    return kwargs_dict

def _get_kwargs(args,**defaults):
    args = args if exists(args) else dict()
    return set_default_kwargs(args,**defaults)


def _select_timerange_build_index(n: int, k1=None, k2=None):
    """
    Build an index array selecting [:k1] and [k2:] from an array of length n.
    """
    if k1 is None and k2 is None:
        return numpy.arange(n)
    if k1 is None:
        return numpy.arange(k2, n)
    if k2 is None:
        return numpy.arange(0, k1)
    return numpy.r_[0:k1, k2:n]

def _select_timerange_apply_index(arrays, idx):
    """
    Apply precomputed indices idx to a list of arrays.
    Always returns contiguous arrays.
    """
    return [numpy.ascontiguousarray(a[idx]) for a in arrays]

def find_closest_value(X, x_c, return_values=False, return_all=False):
    """
    Find the index or value in an array closest to a target or set of targets.

    Parameters
    ----------
    X : array-like
        The array of values to search within.
    x_c : float or array-like
        The target value(s) to find the closest match for.
    return_values : bool, optional
        If True, return the actual values from `X` closest to `x_c`.
        If False, return the indices of the closest values.
        This parameter is ignored if return_all is True.
        Default is False.
    return_all : bool, optional
        If True, returns a tuple (indices,values) with both indices and values.
        If True, ignores return_values.
        Default is False.

    Returns
    -------
        If 'return_all' == True:
            (values,indices)
        Otherwise:
            values  -> if 'return_values' == True
            indices -> if 'return_values' == False

    Notes
    -----
    - Works with both scalar and array inputs for `x_c`.
    - Uses absolute difference to determine closeness.
    - Returns the first closest match if multiple values are equally close.
    """
    X   = numpy.asarray(X)
    x_c = numpy.atleast_1d(x_c)

    # Find index of closest value
    closest_value_internal = lambda X, target: numpy.argmin(numpy.abs(X - target))

    # Apply to each target value
    indices = _make_scalar([closest_value_internal(X, target) for target in x_c])

    if return_all:
        return (X[indices],indices)
    else:
        return X[indices] if return_values else indices

def _is_list_of_str(X):
    return (type(X) is list) and ((len(X)>0) and (type(X[0]) is str))

def get_par_value_from_str(str_list,par_name,dtype=int):
    if type(str_list) is str:
        return dtype(match.group(1)) if (match := re.search(par_name + r'(\d+(?:\.\d+)?)', str_list)) else None
    else:
        return [get_par_value_from_str(idl,par_name,dtype) for idl in str_list]

def par_value_in_str(str_list,par_name,par_value,dtype=int,par_str_fmt=None):
    """
    par_str_fmt == '{:FLAG}'
                where FLAG is, e.g.,
                d   for integers
                g   for generic (autoformat numbers)
                .5f for 5 decimal places floating point number
    """
    if type(str_list) is str:
        par_value = numpy.atleast_1d(par_value).flatten()
        if exists(par_str_fmt):
            par_value = [ par_str_fmt.format(v) for v in par_value ]
            dtype     = str
        val = get_par_value_from_str(str_list,par_name,dtype)
        return val in par_value if exists(val) else False
    else:
        return [ par_value_in_str(st,par_name,par_value,dtype,par_str_fmt) for st in str_list ]

def sort_str_by_par_value(str_list,par_name,dtype=int):
    if _is_list_of_str(str_list):
        return sorted(str_list, key=lambda s: get_par_value_from_str(s,par_name,dtype))
    elif type(str_list) is list:
        return [ sort_str_by_par_value(s,par_name,dtype) for s in str_list ]
    else:
        return str_list

def select_str_by_par_value(str_list,par_name,par_value,dtype=int,par_str_fmt=None,sort_result=True):
    """
    par_str_fmt == '{:FLAG}'
                where FLAG is, e.g.,
                d   for integers
                g   for generic (autoformat numbers)
                .5f for 5 decimal places floating point number
    """
    if _is_list_of_str(str_list):
        if sort_result:
            return sort_str_by_par_value(select_str_by_par_value(str_list,par_name,par_value,dtype,par_str_fmt,sort_result=False),par_name,dtype)
        else:
            return [ s for s in str_list if par_value_in_str(s,par_name,par_value,dtype,par_str_fmt)   ]
    elif type(str_list) is list:
        return [ select_str_by_par_value(s,par_name,par_value,dtype,par_str_fmt,sort_result) for s in str_list ]
    else:
        return str_list
        

def txt_in_str(str_list,txt_list,condition_func=all):
    """
    checks if all or any of the strings in txt_list are in any of the strings in str_list
    condition_func is either all or any

    returns:
        if str_list is a string:
            True if txt_list in str_list according to condition_func (either all or any of the txt_list elements)
            False otherwise
        if str_list is a list of strings:
            a list testing whether each element in str_list contains txt_list according to condition_func
    """
    if type(str_list) is str:
        return condition_func(txt in str_list for txt in txt_list)
    else:
        return [ txt_in_str(st,txt_list,condition_func) in st for st in str_list]

def get_percent_load_str(k,n_tot):
    load_percent = 100*(k+1)//n_tot
    return f'{k+1}/{n_tot} ({load_percent}%)'

def nonempty_str(X):
    return (type(X) is str) and (len(X)>0)

#@njit([
#    types.optional(types.int64)(types.float64[:],types.float64),
#    types.optional(types.int64)(types.int64[:],types.int64)
#])
@njit
def find_first(X,v):
    for k,x in enumerate(X):
        if x == v:
            return k
    return None

#@njit([
#    types.optional(types.int64)(types.float64[:],types.float64),
#    types.optional(types.int64)(types.int64[:],types.int64)
#])
@njit
def find_last(X,v):
    for k in range(len(X)-1,-1,-1):
        if X[k] == v:
            return k
    return None

#def find_first(X,v):
#    k = numpy.argmax(X==v)
#    return k if X[k]==v else None

#def find_last(X,v):
#    k = X.size - 1 - numpy.argmax(X[::-1]==v)
#    return k if X[k]==v else None

def unpack_list_of_tuples(lst):
    """
    given a list of len m where element is an n-tuple, then returns an n-tuple where each element containins m elements
    (similar to numpy.transpose)

    this is useful to unpack a list comprehension applied to a function that has multiple returns
    """
    return tuple( list(x) for x in zip(*lst) )

#Para separar seguimentos contínuos de dados true
def find_contiguous_pieces(cond):
    """
    finds all the sequential cond==True that are separated by at least one cond==False
    
    returns a list where each entry contains
            the indices where cond has sequential True's
    """
    if len(cond) == 0:
        return []
    f = numpy.logical_xor(cond[:-1],cond[1:])
    k_start = numpy.nonzero(numpy.logical_and( f , cond[1:]  ))[0] + 1 # index of the start of a piece
    k_end   = numpy.nonzero(numpy.logical_and( f , cond[:-1] ))[0] + 1 # index of the end of a piece
    if cond[0]:
        k_start = numpy.insert(k_start,0,0)
    if cond[-1]:
        k_end = numpy.insert(k_end,k_end.size,len(cond))
    if (k_start.size > 0) and (k_end.size > 0):
        return [ numpy.arange(a,b) for a,b in zip(k_start,k_end) ]
    return []

def _is_numpy_array(x):
    return type(x) is numpy.ndarray

def exists(x):
    return not(type(x) is type(None))

def _sort_tuple_list(l,k):
    return [ l[i] for i in k ]

def _flatten_tuple_list(l):
    r = []
    for e in l:
        if type(e) is list:
            r += _flatten_tuple_list(e)
        else:
            r.append(e)
    return r

def _get_unique_pair_indices(k):
    if not (type(k) is list):
        _,a = numpy.unique(k, axis=0, return_inverse=True)
        return k
    n = len(k)
    a = numpy.array(k).flatten()
    _,a = numpy.unique(a, axis=0, return_inverse=True)
    return _split_array(a,n)

def _split_array(a, n):
    k, m = divmod(len(a), n)
    return numpy.array([ a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n) ])

def sort_A_to_B(A,B,return_ind=False):
    """
    sorts A according to B
    returns A, the indices that sort A [[ A[ind] == A_sorted  ]]
    """
    map_AB    = { b:k for k,b in enumerate(B) }
    ind,A_srt = unpack_list_of_tuples(sorted(list(enumerate(A)),key=lambda pair:map_AB[pair[1]]))
    if _is_numpy_array(A):
        A_srt = numpy.array(A_srt)
    if return_ind:
        return A_srt,ind
    else:
        return A_srt

def smooth_data(y,filter_func,filter_args,y_outlier_mask_args=None,x=None):
    y_outlier_mask_args                         = _get_kwargs(y_outlier_mask_args, window=20, z_score_local_threshold=3,x=x)
    y_outlier_mask_args['return_filtered_data'] = False
    mask_y                                      = outlier_mask_local_filter(y,**y_outlier_mask_args)
    y_flt                                       = filter_func(y[mask_y],**filter_args)
    if exists(x):
        return x[mask_y],y_flt
    else:
        return y_flt

def outlier_mask_local_filter(y, window=20, z_score_local_threshold=3,return_filtered_data=False,x=None):
    y    = numpy.asarray(y)
    mask = numpy.ones_like(y, dtype=bool)
    for i in range(len(y)):
        w_start      = max(0, i - window // 2)
        w_end        = min(len(y), i + window // 2)
        y_local      = y[w_start:w_end]
        y_local_mean = numpy.nanmean(y_local)
        y_local_std  = numpy.nanstd( y_local)
        if abs(y[i] - y_local_mean) > z_score_local_threshold * y_local_std:
            mask[i] = False
    if return_filtered_data:
        if exists(x):
            x = numpy.asarray(x)
            return x[mask],y[mask]
        return y[mask]
    else:
        return mask

def linearized_fit(x_data, y_data, x_transform=None, y_transform=None, y_inverse_transform=None, mask=None, slope=None, verbose=True):
    """
    Performs linear regression on transformed data using scipy.stats.linregress.
    If 'slope' is provided, keeps the slope fixed and fits only the intercept.
    
    I.e., this function fits a line
        y_transform(y_data) = slope*x_transform(x_data) + intercept
    and returns
        fitpar = (slope, intercept)
    
    Parameters:
    ----------
    x_data : array-like
        Original independent variable values.
    y_data : array-like
        Original dependent variable values.
    x_transform : callable
        Function to transform x_data for linearization.
    y_transform : callable
        Function to transform y_data for linearization.
    y_inverse_transform : callable
        Function to reverse the y_transform for plotting fitted curve in original space.
    mask : numpy-compatible index [e.g., range, int, logical, etc]
        if present, only performs fit on the masked data: x_data[mask], y_data[mask]
    slope : float or None
        If provided, fixes the slope and fits only the intercept.

    Returns:
    -------
    fitpar : tuple
        (slope, intercept) of the linear fit in transformed space.
    r_squared : float
        Coefficient of determination (R²).
    residual_sum_squares : float
        Residual sum of squares in original space.
    x_fit : ndarray
        Dense x values for plotting the fitted curve.
    y_fit : ndarray
        Fitted y values in original space.
    """
    if not exists(x_transform):
        x_transform = lambda x:x
    if not exists(y_transform):
        y_transform         = lambda y:y
        y_inverse_transform = lambda y:y
    assert exists(y_transform) and exists(y_inverse_transform), "y_inverse_transform must be set when y_transform is set"
    
    # func to fit
    func  = lambda x,*fitpar: y_inverse_transform(fitpar[0] * x_transform(x) + fitpar[1])

    # Transform the data
    if exists(mask):
        x_lin = x_transform(x_data[mask])
        y_lin = y_transform(y_data[mask])
    else:
        x_lin = x_transform(x_data)
        y_lin = y_transform(y_data)

    if len(x_lin) < 2:
        if verbose:
            warnings.warn('Only 2 data points, so returning empty fit...')
        return structtype(func=func,fitpar=(numpy.nan, numpy.nan),R2=numpy.nan,res_sum_sqr=numpy.nan,x_fit=x_lin,y_fit=y_lin)

    # Perform linear regression
    if exists(slope):
        # Fit only the intercept (least-squares estimate for fixed slope)
        # intercept = mean(y - slope*x)
        intercept = numpy.nanmean(y_lin - slope * x_lin)
        # Compute correlation coefficient manually for R²
        y_fit_lin = slope * x_lin + intercept
        ss_res    = numpy.nansum((y_lin - y_fit_lin) ** 2)
        ss_tot    = numpy.nansum((y_lin - numpy.nanmean(y_lin)) ** 2)
        r_value   = numpy.sqrt(1 - ss_res / ss_tot) if ss_tot > 0 else numpy.nan
    else:
        # Perform linear regression normally
        slope, intercept, r_value, _, _ = scipy.stats.linregress(x_lin, y_lin)

    # Generate fitted values in original space
    x_fit = numpy.linspace(min(x_data), max(x_data), 100)
    y_fit = func(x_fit,slope,intercept)

    # Compute residuals and RSS in original space
    y_pred    = y_inverse_transform(slope * x_lin + intercept)
    residuals = (y_data[mask] if exists(mask) else y_data) - y_pred
    rss       = numpy.sum(residuals**2)

    return structtype(func=func,fitpar=(slope, intercept),R2=r_value**2,res_sum_sqr=rss,x_fit=x_fit,y_fit=y_fit)

def _get_range_corrected_boundary(a,b,N):
    if b-a > N:
        return range(N)
    if a < 0:
        a,b = 0,b+abs(a)
    if b > N:
        a,b = a-abs(N-b),N
    return range(a,b)

def linearized_fit_local(n_points,**linearized_fit_args):
    """
    performs fits of n_points throughout the whole x-domain using linearized_fit
    """
    x_data          = linearized_fit_args.pop('x_data')
    y_data          = linearized_fit_args.pop('y_data')
    mask            = linearized_fit_args.pop('mask',None)
    n_data          = len(x_data)
    data_fits_range = range(n_points//2,(0 if n_points%2 else 1) + n_data - n_points//2)
    fr              = get_empty_list(len(data_fits_range))
    ind_list        = get_empty_list(len(data_fits_range))
    n               = -1
    for k in data_fits_range:
        n         += 1
        ind        = _get_range_corrected_boundary(k-n_points//2,(1 if n_points%2 else 0)+k+n_points//2,n_data)
        x          = x_data[ind]
        y          = y_data[ind]
        if exists(mask):
            mask_local = mask[ind]
        ind_list[n]    = ind
        fr[n]          = linearized_fit(x,y,mask=mask_local,**linearized_fit_args)
    return fr,ind_list


type_statistic_func = types.FunctionType(types.float64(types.float64[:]))
@njit(types.float64[:](types.float64[:],type_statistic_func,types.int64,types.int64))
def bootstrap_with_resample_size(data, statistic_func, n_resamples, resample_size):
    """
    Performs bootstrap resampling with a custom resample size using Numba.

    This function is a high-performance alternative to scipy's bootstrap
    for cases where a different resample size is needed.

    Args:
    data (np.ndarray): The 1D NumPy array of input data.
    statistic_func (numba.jit.numba.core.dispatcher.Dispatcher): A Numba-jitted
        function that takes a 1D NumPy array and returns a single float.
    n_resamples (int): The number of bootstrap iterations to perform.
    resample_size (int): The size of each bootstrap sample.
    random_state (int, optional): An integer seed for the random number
        generator to ensure reproducibility. Defaults to -1 (no seed).

    Returns:
    np.ndarray: A 1D NumPy array containing the statistic value for each
        of the `n_resamples` iterations.
    """
    bootstrap_distribution = numpy.empty(n_resamples, dtype=data.dtype)
    N                      = data.size
    for k in range(n_resamples):
        resample_indices          = numpy.random.choice(N, size=resample_size, replace=True)
        bootstrap_distribution[k] = statistic_func(data[resample_indices])
    return bootstrap_distribution

@njit(types.float64(types.float64[:]))
def my_stddev(X):
    return numpy.nanstd(X)

@njit(types.float64(types.float64[:]))
def my_variance(X):
    return numpy.nanvar(X)


def bootstrap_func(x,func,return_bs_confint_se=False,**bootstrapArgs):
    """
    this function repeatedly (n_resamples times) applies func(x) along the specified axis (if any)
    and then returns the mean value of func(x) over these n_resamples samples

    delegate parameters to scipy.stats.bootstrap:
        axis             -> axis along which to apply func
        n_resamples      -> number of times we draw a sample from x along axis
        vectorized       -> parallel computations
        confidence_level -> confidence interval within which we calculate func(x)

    example:
    >>> x = 10*numpy.random.randn(1000) # stddev of x == 10
    >>> bootstrap_func(x,numpy.std) # returns approx 10
    """
    if not(type(x) is numpy.ndarray):
        x         = numpy.asarray(x)
    bootstrapArgs = set_default_kwargs(bootstrapArgs,n_resamples=10,vectorized=True,confidence_level=0.95)
    is_1d         = False
    if x.ndim == 1:
        is_1d                 = True
        x                     = x.reshape((x.size,1))
        bootstrapArgs['axis'] = 0
    bs       = scipy.stats.bootstrap((x,),func,**bootstrapArgs)
    get_elem = lambda el: el[0] if is_1d else el
    m        = (bs.confidence_interval.low + bs.confidence_interval.high)/2.0
    m        = get_elem(m)
    confint  = (get_elem(bs.confidence_interval.low), get_elem(bs.confidence_interval.high))
    s        = get_elem(bs.standard_error)
    result   = m
    if return_bs_confint_se:
        result = (m,confint,s)
    return result


class structtype(collections.abc.MutableMapping):
    def __init__(self,struct_fields=None,field_values=None,**kwargs):
        if not(type(struct_fields) is type(None)):
            #assert not(type(values) is type(None)),"if you provide field names, you must provide field values"
            if not self._is_iterable(struct_fields):
                struct_fields = [struct_fields]
                field_values  = [field_values]
            kwargs.update({f:v for f,v in zip(struct_fields,field_values)})
        self.Set(**kwargs)
    def Set(self,**kwargs):
        self.__dict__.update(kwargs)
        return self
    def SetAttr(self,field,value):
        if not self._is_iterable(field):
            field = [field]
            value = [value]
        self.__dict__.update({f:v for f,v in zip(field,value)})
        return self
    def GetFields(self,sep='; '):
        return sep.join([ k for k in self.__dict__.keys() if (k[0:2] != '__') and (k[-2:] != '__') ])
        #return self.__dict__.keys()
    def IsField(self,field):
        return field in self.__dict__.keys()
    def RemoveField(self,field):
        return self.__dict__.pop(field,None)
    def RemoveFields(self,*fields):
        r = []
        for k in fields:
            r.append(self.__dict__.pop(k,None))
        return r
    def KeepFields(self,*fields):
        keys = list(self.__dict__.keys())
        for k in keys:
            if not (k in fields):
                self.__dict__.pop(k,None)
    def keys(self):
        return self.__dict__.keys()
    def items(self):
        return self.__dict__.items()
    def values(self):
        return self.__dict__.values()
    def pop(self,key,default_value=None):
        if type(key) is str:
            return self.__dict__.pop(key,default_value)
        elif isinstance(key,collections.abc.Iterable):
            r = []
            for k in key:
                r.append(self.__dict__.pop(k,default_value))
            return r
        else:
            raise ValueError('key must be a string or a list of strings')
    def __setitem__(self,label,value):
        self.__dict__[label] = value
    def __getitem__(self,label):
        return self.__dict__[label]
    def __repr__(self):
        char_lim      = 50
        char_arg_name = 20
        get_repr      = lambda r: r[:char_lim]+'...' if len(r) > char_lim else r
        type_name     = type(self).__name__
        arg_strings   = []
        star_args     = {}
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_name     = (name[:(char_arg_name-3)] + '...') if len(name) > char_arg_name else name.rjust(char_arg_name)
                arg_strings.append('%s: %s' % (arg_name, get_repr(repr(value)).replace('\n','').strip()  ))
            else:
                star_args[name] = get_repr(repr(value))
        if star_args:
            arg_strings.append('**%s' % star_args)
        sep = '\n' #if len(arg_strings) > 3 else '; '
        return '%s(\n%s\n)' % (type_name, sep.join(arg_strings))
    def _get_kwargs(self):
        return sorted(self.__dict__.items())
    def _get_args(self):
        return []
    def _is_iterable(self,obj):
        return (type(obj) is list) or (type(obj) is tuple)
    def __delitem__(self,*args):
        self.__dict__.__delitem__(*args)
    def __len__(self):
        return self.__dict__.__len__()
    def __iter__(self):
        return iter(self.__dict__)
