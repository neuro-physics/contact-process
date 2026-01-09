import numpy
from enum import IntEnum
from numba import njit,types,typeof
from numba.typed import List
from modules.cpsim_io_options import is_parallel_update,Options_graph,Options_iterdynamics,Options_sim


def Get_Simulation_Func(args):
    #if args.sim == SimulationType.AVAL:
    #    if not is_parallel_update(args.update):
    #        args.update     = UpdateType.PARALLEL # forcing parallel update for avalanche
    #        args.expandtime = False
    #        print(' ::: WARNING ::: forcing parallel update and no expandtime because sim == %s'%args.sim)
    if args.graph == Options_graph.ALLTOALL:
        if is_parallel_update(args.update):
            return Run_MF_parallel
        else:
            return Run_MF_sequential
    else: # (args.graph == 'ring') or (args.graph == 'ringfree')
        if is_parallel_update(args.update):
            return Run_Graph_parallel
        else:
            return Run_Graph_sequential

def Get_Simulation_Timescale(args):
    dt = 1.0
    if args.expandtime and (not is_parallel_update(args.update)):
        dt = 1.0 / float(args.N) # this time scale is suggested in Dickman and Marro book; Henkel normalizes the time scale by total rate too -- pg 87 pdf, paragraph after eq. 3.35 book Henkel
    else:
        if args.expandtime:
            print(' ::: WARNING ::: ignoring expandtime')
    return dt


#@njit
#def random_sample(N, K):
#    # Initialize a NumPy array with the range of values from 0 to N-1
#    population = numpy.arange(N, dtype=numpy.int64)
#    
#    # Perform Fisher-Yates shuffle for the first K elements
#    for i in range(K):
#        # Pick a random index from i to N-1
#        j = random.randint(i, N-1)
#        # Swap the elements at indices i and j
#        population[i], population[j] = population[j], population[i]
#    
#    # Return the first K elements as a NumPy array
#    return population[:K]

#@njit
#def get_random_state(X,f_act):
#    # X -> site vector (in/out parameter); numpy.ndarray
#    # f_act -> fraction of active elements
#    ind = random_sample(len(X), int(f_act * len(X)))
#    for i in ind:
#        X[i] = 1.0


@njit(types.int64[:](types.int64[:],types.float64))
def get_ordered_state(X,f_act):
    N = len(X)
    K = int(f_act * N)
    for i in range(K):
        X[i] = 1
    for i in range(K,N):
        X[i] = 0
    return X

@njit(types.int64[:](types.int64[:],types.float64))
def get_random_state(X,f_act):
    # X -> site vector (in/out parameter); numpy.ndarray
    # f_act -> fraction of active elements
    N = len(X)
    X = get_ordered_state(X,f_act)
    for i in range(N-1):
        #j = random.randint(i, N-1)
        j = numpy.random.randint(i, N)
        X[i],X[j] = X[j],X[i] # Shuffle X in-place using Fisher-Yates
    return X

_type_X_data_item = types.Tuple((types.float64, types.int64, types.int64))
_type_X_data      = types.ListType(_type_X_data_item)

@njit(_type_X_data(_type_X_data,types.float64,types.int64,types.int64))
def save_spk_data_fake(X_data, t, k, X):
    return X_data

@njit(_type_X_data(_type_X_data,types.float64,types.int64,types.int64))
def save_spk_data(X_data, t, k, X):
    if X:
        X_data.append((t,k,X))
        #X_values.append(X)
        #X_ind.append(k)
        #X_time.append(t)
    return X_data

@njit(_type_X_data(_type_X_data,types.float64,types.int64,types.int64))
def write_spk_data_fake(X_data,t,k,X):
    return X_data

@njit(_type_X_data(_type_X_data,types.float64,types.int64,types.int64))
def write_spk_data(X_data,t,k,X):
    if X:
        #spkFile.write(str(t) + ',' + str(k) + ',' + str(X) + '\n')
        print(t,',',k,',',X)
    return X_data

@njit(_type_X_data(_type_X_data,types.float64,types.int64,types.int64))
def write_spk_data_debug(X_data,t,k,X):
    if X:
        #spkFile.write(str(t) + ',' + str(k) + ',' + str(X) + '\n')
        print(t,',',k,',',X,'#debug')
    return X_data


_type_writesave_spk_data = types.FunctionType(_type_X_data(_type_X_data,types.float64,types.int64,types.int64))
@njit(types.Tuple((_type_writesave_spk_data, _type_writesave_spk_data))(types.boolean, types.boolean))
def get_write_spike_data_functions(saveSites,writeOnRun):
    if saveSites:
        if writeOnRun:
            write_spk_time = write_spk_data #lambda t_ind,k_ind: spkTimeFile.write(str(t_ind) + ',' + str(k_ind) + '\n')
            save_spk_time  = save_spk_data_fake
        else:
            write_spk_time = write_spk_data_fake
            save_spk_time  = save_spk_data
    else:
        write_spk_time = write_spk_data_fake
        save_spk_time  = save_spk_data_fake
    return write_spk_time,save_spk_time

@njit(_type_X_data(types.int64[:],types.float64,types.boolean,types.boolean))
def save_initial_network_state(X, t0, saveSites, writeOnRun):
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_data                       = List.empty_list(_type_X_data_item) # get_initial_network_state_for_output(X,saveSites and not writeOnRun)
    N                            = len(X)
    for i in range(N):
        X_data = save_spk_time( X_data, t0, i, X[i]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        _      = write_spk_time(X_data, t0, i, X[i]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
    return X_data

@njit(_type_X_data(types.boolean,_type_writesave_spk_data,_type_writesave_spk_data,_type_X_data,types.boolean,types.float64,types.float64,types.int64,types.int64[:],types.int64))
def dump_spike_data_sequential_update(do_dump_data,save_spk_time_func,write_spk_time_func,X_data,dtsample_is_1,t,dt,i,X,Xa):
    if do_dump_data:
        for k in range(X.size):
            X_data  = save_spk_time_func( X_data, t*dt, k, X[k]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
            _       = write_spk_time_func(X_data, t*dt, k, X[k]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
    elif dtsample_is_1:
        X_data  = save_spk_time_func( X_data, t*dt, i, X[i]-Xa) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        _       = write_spk_time_func(X_data, t*dt, i, X[i]-Xa) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
    return X_data

@njit(types.none(types.string,types.boolean,types.boolean))
def open_file(spkFileName,saveSites_and_writeOnRun,save_site_state_change):
    if saveSites_and_writeOnRun:
        #spk_file = open(spkFileName,'w')
        #spk_file.write('#t,k,Xk\n') # header
        #print('spk file opened: %s'%spkFileName)
        #return spk_file
        #print('*** writing file ',spkFileName,' during simulation')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('################################################## file ', spkFileName, ' will be printed to stdout')
        #print('################################################## due to numba limitation')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('[[[ BEGINNING OF FILE ]]]')
        if save_site_state_change:
            header_txt = '#t,k,dX'
        else:
            header_txt = '#t,k,X'
        print(header_txt)
    return None

@njit(types.void(types.none,types.string,types.boolean))
def close_file(spkFile,spkFileName,saveSites_and_writeOnRun):
    if saveSites_and_writeOnRun:
        print('')
        #spkFile.close()
        #print('spk file close')
        #print('[[[ END OF FILE ]]]')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')



@njit(types.Tuple((types.int64,types.int64))(types.int64))
def _largest_close_factors(N):
    sqrt_N = int(numpy.sqrt(N))
    for i in range(sqrt_N, 0, -1):
        if N % i == 0:
            return i, N // i
    return 1, N  # fallback for prime N

_get_neighbors_output_type = types.Tuple((types.int64[:,:],types.int64[:]))
@njit(_get_neighbors_output_type(types.int64))
def get_ring_neighbors_periodic(N):
    n = numpy.empty((N, 2), dtype=numpy.int64) # index of neighbors
    K = 2*numpy.ones(N,dtype=numpy.int64)      # number of neighbors
    for k in range(N):
        n[k, 0] = (k - 1) % N  # left neighbor
        n[k, 1] = (k + 1) % N  # right neighbor
    return n,K

@njit(_get_neighbors_output_type(types.int64))
def get_ring_neighbors_free(N):
    n,K      = get_ring_neighbors_periodic(N) # n=index of neighbors ; K=number of neighbors
    n[0  ,0] = 1    # first site connects only to the right
    n[0  ,1] = -666 # first site connects only to the right
    n[N-1,0] = N-2  # last site connects only to the left
    n[N-1,1] = -666 # last site connects only to the left
    K[0]     = 1 # first site has only 1 neighbor
    K[N-1]   = 1 # last site has only 1 neighbor
    return n,K

@njit(_get_neighbors_output_type(types.int64))
def get_squareperiodic_neighbors(N):
    Ly,Lx = _largest_close_factors(N) # Lx -> number of columns; Ly -> number of rows
    n     = numpy.empty((N, 4), dtype=numpy.int64) # index of neighbors
    K     = 4*numpy.ones(N,dtype=numpy.int64)      # number of neighbors
    for y in range(Ly): # rows
        for x in range(Lx): # columns
            k       = x + y*Lx
            n[k, 0] = (((x-1)%Lx) +     y      * Lx)  # left neighbor
            n[k, 1] = (((x+1)%Lx) +     y      * Lx)  # right neighbor
            n[k, 2] = (    x      + ((y-1)%Ly) * Lx)  # up neighbor    ---  x grows from top to bottom
            n[k, 3] = (    x      + ((y+1)%Ly) * Lx)  # down neighbor  ---  x grows from top to bottom
    return n,K

@njit(_get_neighbors_output_type(types.int64))
def get_squarefree_neighbors(N):
    Ly,Lx = _largest_close_factors(N) # Lx -> number of columns; Ly -> number of rows
    n     = -666*numpy.ones((N, 4), dtype=numpy.int64) # index of neighbors
    K     =    4*numpy.ones(N,dtype=numpy.int64)       # number of neighbors
    for y in range(Ly):
        for x in range(Lx):
            k   = x + y*Lx
            loc = []
            Kl  = 0
            k_l = (((x-1)%Lx) +     y      * Lx)
            k_r = (((x+1)%Lx) +     y      * Lx)
            k_u = (    x      + ((y-1)%Ly) * Lx)
            k_d = (    x      + ((y+1)%Ly) * Lx)
            if y > 0: # not top row
                loc.append(k_u) # has an upper neighbor
                Kl+=1
            if y < (Ly-1): # not bottom row
                loc.append(k_d) # has a down neighbor
                Kl+=1
            if x > 0: # not left column
                loc.append(k_l) # has a left neighbor
                Kl+=1
            if x < (Lx-1): # not right column
                loc.append(k_r) # has a right neighbor
                Kl+=1
            n[k,:Kl] = loc
            K[k]     = Kl
    return n,K

@njit(_get_neighbors_output_type(types.int64,types.int64))
def get_graph_neighbors(graph:Options_graph,N):
    if graph == Options_graph.RING:
        return get_ring_neighbors_periodic(N)
    elif graph == Options_graph.RINGFREE:
        return get_ring_neighbors_free(N)
    elif graph == Options_graph.SQUAREPERIODIC:
        return get_squareperiodic_neighbors(N)
    elif graph == Options_graph.SQUAREFREE:
        return get_squarefree_neighbors(N)
    else:
        raise ValueError(f'get_neighbors not defined for graph {graph}')

@njit(types.int64(types.boolean))
def bool2int(x):
    return 1 if x else 0

@njit(types.int64(types.int64,types.float64,types.float64))
def state_iter_Tome_Oliveira(X,n,inv_l):
    """
    The description in the book has a typo... It was supposed to be 1/(lambda + 1) instead of 1/lambda...
    However, the code is kept here for reference. The corrected code is in state_iter_Tome_Oliveira_mod

     described in pg 308pdf/402 Tome Oliveira book before eq 13.6
     At each time step we choose a site at random, say site i.
       (a) If i is occupied, than we generate a random number r uniformly distributed in the interval [0;1].
           If r <= 1/lambda = inv_l, the particle is annihilated and the site becomes empty.
           Otherwise, the site remains occupied.
       (b) If i is empty, then one of its neighbors is chosen at random.
           If the neighboring site is occupied then we create a particle at site i.
           Otherwise, the site i remains empty. 
    
     X -> state of node
     n -> fraction of active neighbors of X
     inv_l -> inverse of activation rate: inv_l = 1/lambda = alpha in the book
    
     This algorithm generates, at each time step, a probability of creation Pc:
     Pc = P[ Xi(t+1)=1 and Xi(t)=0 ] = P[Xi(t)=0] * P[r<n]        = (1-rho) * n  [[[ rho -> fraction of active sites in total ]]]
     and a probability of annihilation Pa:
     Pa = P[ Xi(t+1)=0 and Xi(t)=1 ] = P[Xi(t)=1] * P[r<1/lambda] = rho / lambda
     these values contrast with Pc and Pa from the Dickman algorithm (see state_iter_Dickman)
     and end-up generating a different critical point lambda_c ~ 2  [[[ periodic ring, sequential update, Dickman's lambda_c ~ 3.3 ]]]
    
     returns the new state based on the previous state X for a given node
             site is occupied, so it eliminates               site is empty, so it creates
             the particle with prob 1/lambda                 the particle with the same chance as that of finding an active neighbor
     """
    #return bool2int( numpy.random.random() > inv_l ) if X else bool2int(numpy.random.random() < n)
    if X: # site is occupied
        return bool2int(numpy.random.random() > inv_l) # r > 1/lambda: stays occupied; r < 1/lambda: annihilation
    else: # site is empty
        return bool2int(numpy.random.random() < n) # r < n: infection; r>n: stays empty

@njit(types.int64(types.int64,types.float64,types.float64))
def state_iter_Tome_Oliveira_mod(X,n,inv_l):
    """
     MODIFIED TO MATCH THE DICKMAN algorithm
     described in pg 308pdf/402 Tome Oliveira book before eq 13.6
     also matches the description in pg 77 (87 of pdf) of the Henkel-Hinrichsen-Lubeck book
     [probabilities given after Eq. 3.35].
     Tome-Oliveira description
     [I believe they meant 1/(1+lambda) instead of 1/lambda;
     and also, creation only if random neighbor is active AND probability lambda/(1+lambda)]:
     At each time step we choose a site at random, say site i.
       (a) If i is occupied, than we generate a random number r uniformly distributed in the interval [0;1].
           If r <= 1/lambda = inv_l, the particle is annihilated and the site becomes empty.
           Otherwise, the site remains occupied.
       (b) If i is empty, then one of its neighbors is chosen at random.
           If the neighboring site is occupied then we create a particle at site i.
           Otherwise, the site i remains empty. 
    
     X -> state of node
     n -> fraction of active neighbors of X
     inv_l -> inverse of activation rate: inv_l = 1/lambda = alpha in the book
    
     This algorithm generates, at each time step, a probability of creation Pc:
     Pc = P[ Xi(t+1)=1 and Xi(t)=0 ] = P[Xi(t)=0] * P[r<n]        = (1-rho) * n  [[[ rho -> fraction of active sites in total ]]]
     and a probability of annihilation Pa:
     Pa = P[ Xi(t+1)=0 and Xi(t)=1 ] = P[Xi(t)=1] * P[r<1/lambda] = rho / lambda
     these values contrast with Pc and Pa from the Dickman algorithm (see state_iter_Dickman)
     and end-up generating a different critical point lambda_c ~ 2  [[[ periodic ring, sequential update, Dickman's lambda_c ~ 3.3 ]]]
    
     returns the new state based on the previous state X for a given node
             site is occupied, so it eliminates               site is empty, so it creates
             the particle with prob 1/lambda                 the particle with the same chance as that of finding an active neighbor
    """
    #return bool2int( numpy.random.random() > inv_l ) if X else bool2int(numpy.random.random() < n)
    v = 1.0/(1.0 + inv_l) # v = lambda / (1 + lambda)
    if X: # site is occupied
        return bool2int(numpy.random.random() > inv_l*v) # r > 1/(1+lambda): stays occupied; r < 1/(1+lambda): annihilation
    else: # site is empty
        return bool2int(numpy.random.random() < n*v) # r < n*lambda / (1 + lambda): infection; r>n: stays empty

@njit(types.int64(types.int64,types.float64,types.float64))
def state_iter_Dickman_mod(X,n,v):
    """
    SEEMS TO HAVE WRONG RATES (check debug table of transition rates)
    kept here only for reference

     this code was adapted from what was
     described in pg 178pdf/162book Marro & Dickman book.
     Each step involves randomly choosing a process - creation with probability v=lambda/(1+lambda),
     annihilation with probability 1-v -- and a lattice site x.
     In an annihilation event, the particle (if any) at x is removed.
     Creation proceeds only if x is occupied and a randomly chosen nearest-neighbor y is vacant;
     if so, a new particle is placed at y.
     Time is incremented by At after each step, successful or not.
     (Normally one takes Delta t = 1/N on a lattice of N sites, so that a unit time interval,
     or MC step, corresponds, on average, to one attempted event per site.)
    
     X -> state of node
     n -> fraction of active neighbors of X
     v -> lambda / (1+lambda) (creation event probability)
    
     This algorithm generates, at each time step, a probability of creation Pc:
     Pc = P[ e=c and Xi(t)=1 and Xj(t)=0 ] = P[e=c] * P[Xi(t)=1] * P[Xj(t)=0] = v * rho * (1-n)  [[[ rho -> fraction of active sites in total; e=event (c or a) ]]]
          I assume, I can invert the order of neighbor and selected site, so that
     Pc = v * (1-rho) * n  [[[ i.e., current selected site is inactive and there is an active neighbor ]]]
     and a probability of annihilation Pa:
     Pa = P[ e=a and Xi(t)=1 ]             = P[e=a] * P[Xi(t)=1]              = (1-v)*rho
     these values contrast with Pc and Pa from the Tome-Oliveira algorithm (see state_iter_Tome_Oliveira)
     and generate (hopefully) lambda_c ~ 3.3  [[[ periodic ring, sequential update ]]]
    
     returns the new state based on the previous state X for a given node
             site is occupied, so it eliminates               site is empty, so it creates
             the particle with prob 1/lambda                 the particle with the same chance as that of finding an active neighbor
    v = 1.0 / (1.0 + inv_l) # v === lambda / (1+lambda); but as a function of inv_l = 1/lambda
    """
    if X: # site is occupied
          # Prob = rho
          # then a particle is annihilated with chance (1-v) [[[ hence, r > v; also, implicit is the '*rho' bit in the if condition ]]]
          # otherwise nothing happens (X=1 remains)
        return bool2int(numpy.random.random() > v)
    else: # site is empty
          # Prob = 1 - rho
          # then a particle is created with chance v*n [[[*(1-rho), implicit in the if condition]]]
          # otherwise, nothing happens (X=0 remains)
        return bool2int(numpy.random.random() < v*n)

@njit(types.int64(types.int64,types.float64,types.float64))
def state_iter_Dickman(X, n, v):
    """
     described in pg 178pdf/162book Marro & Dickman book
     Each step involves randomly choosing a process - creation with probability v=lambda/(1+lambda),
     annihilation with probability 1-v -- and a lattice site x.
     In an annihilation event, the particle (if any) at x is removed.
     Creation proceeds only if x is occupied and a randomly chosen nearest-neighbor y is vacant;
     if so, a new particle is placed at y.
     Time is incremented by At after each step, successful or not.
     (Normally one takes Delta t = 1/N on a lattice of N sites, so that a unit time interval,
     or MC step, corresponds, on average, to one attempted event per site.)
    
     X -> state of node
     n -> fraction of active neighbors of X
     v -> lambda / (1+lambda) (creation event probability)
    
     This algorithm generates, at each time step, a probability of creation Pc:
     Pc = P[ e=c and Xi(t)=1 and Xj(t)=0 ] = P[e=c] * P[Xi(t)=1] * P[Xj(t)=0] = v * rho * (1-n)  [[[ rho -> fraction of active sites in total; e=event (c or a) ]]]
          I assume, I can invert the order of neighbor and selected site, so that
     Pc = v * (1-rho) * n  [[[ i.e., current selected site is inactive and there is an active neighbor ]]]
     and a probability of annihilation Pa:
     Pa = P[ e=a and Xi(t)=1 ]             = P[e=a] * P[Xi(t)=1]              = (1-v)*rho
     these values contrast with Pc and Pa from the Tome-Oliveira algorithm (see state_iter_Tome_Oliveira)
     and generate (hopefully) lambda_c ~ 3.3  [[[ periodic ring, sequential update ]]]
    
     returns the new state based on the previous state X for a given node
             site is occupied, so it eliminates               site is empty, so it creates
             the particle with prob 1/lambda                 the particle with the same chance as that of finding an active neighbor
    v = 1.0 / (1.0 + inv_l) # v === lambda / (1+lambda); but as a function of inv_l = 1/lambda
    """
    if numpy.random.random() < v:
        # Birth attempt
        return 1 if ((X == 0) and (numpy.random.random() < n)) else X
    else:
        # Death attempt
        return 0 if X == 1 else X


#type_state_iter = typeof(state_iter_Dickman)
_type_state_iter = types.FunctionType(types.int64(types.int64, types.float64, types.float64))
@njit(_type_state_iter(types.int64))
def get_site_state_iterator(iterdynamics):
    if iterdynamics == Options_iterdynamics.TOME_OLIVEIRA:
        state_iter  = state_iter_Tome_Oliveira_mod
    else:
        state_iter  = state_iter_Dickman
    return state_iter

@njit(types.float64(types.int64, types.float64))
def get_site_state_iterator_alpha(iterdynamics,l):
    if iterdynamics == Options_iterdynamics.TOME_OLIVEIRA:
        alpha       = 1.0 / l # chance of annihilating if site is occupied, book TOme e Oliveira
    else:
        alpha       = l / (1.0 + l) # v, book of Marro & Dickman
    return alpha

#@njit
#def stack_add(stack,k):
#    # adds k to the top of the stack s [i.e., to position s(1) ]
#    # s is a vector with fixed length
#    # stack_add will shift all elements of s to the right, and add k as the first element in s
#    stack = numpy.roll(stack,1)
#    stack[0] = k

#type_Listuint_LIFO = types.Tuple((types.int64[:],types.int64))
#@njit(type_Listuint_LIFO(types.int64))
#def Listuint_LIFO_init(size_max):
#    size = 0
#    lst  = numpy.full(size_max,-1,dtype=numpy.int64)
#    return lst,size
#
#@njit(type_Listuint_LIFO(types.int64[:],types.int64,types.int64))
#def Listuint_LIFO_add(lst,size,value):
#    if size < len(lst):
#        lst[size]  = value
#        size      += 1
#    return lst,size
#
#@njit(types.Tuple((types.int64,types.int64[:],types.int64))(types.int64[:],types.int64))
#def Listuint_LIFO_pop(lst,size):
#    if size > 0:
#        size -= 1
#    return lst[size],lst,size # returns last element
#
#@njit(type_Listuint_LIFO(types.int64[:]))
#def get_active_sites_list(X):
#    lst_active_sites,lst_active_sites_N = Listuint_LIFO_init(len(X))
#    for k in range(len(X)):
#        if X[k]:
#            Listuint_LIFO_add(lst_active_sites,lst_active_sites_N,k)
#    return lst_active_sites,lst_active_sites_N

_type_cyclic_stack_data = types.Tuple((types.float64[:],types.int64))
@njit(_type_cyclic_stack_data(types.int64))
def CyclicStack_Init(maxsize):
    #stack, maxsize, count = cyclic_stack_data
    stack = numpy.full(maxsize, 0.0)
    count = 0
    return stack,count

@njit(_type_cyclic_stack_data(types.float64[:],types.int64,types.int64,types.int64,types.float64))
def CyclicStack_Set(stack, maxsize, count, index, value):
    #stack, maxsize, count = cyclic_stack_data
    stack[index % maxsize] = value
    if count < maxsize:
        count += 1
    return stack,count

@njit(types.float64(types.float64[:],types.int64,types.int64))
def CyclicStack_Get(stack, count, index):
    #stack, maxsize, count = cyclic_stack_data
    return stack[index % count]

@njit(types.float64(types.float64[:],types.int64))
def CyclicStack_GetRandom(stack, count):
    #stack, maxsize, count = cyclic_stack_data
    #return stack[random.randint(0,count-1)]
    return stack[numpy.random.randint(0,count)]

#@njit(types.int64(types.int64))
#def CyclicStack_Len(count):
#    #stack, maxsize, count = cyclic_stack_data
#    return count

@njit(types.Tuple((types.boolean, types.int64[:], types.int64))(types.int64[:], types.boolean, types.int64, types.float64[:], types.int64, types.int64))
def check_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count):
    # returns True if activity must continue
    #         False if activity should die out
    if sum_X < 1: # activity died out
        if is_aval_sim: # if it is a simulation for avalanches
            # we always restart the activity
            sum_X                = 1
            X[int((len(X)-1)/2)] = 1 # seeding the middle of the network
        else:
            # otherwise, we pick a state from the memory
            # only if we have memory states (M>0)
            if M == 0: # absorbing state reached and no memory to restart
                return False, X, sum_X
            X     = get_random_state(X, CyclicStack_GetRandom(rho_memory,cs_count))
            sum_X = sum(X)
    return True, X, sum_X

@njit(types.int64[:](types.int64,types.float64,types.boolean,types.int64))
def get_IC(X0, fX0, X0Rand, N):
    X0 = int(X0)
    X  = numpy.zeros(N,dtype=numpy.int64)
    if X0Rand:
        #X[random.sample(range(N),k=int(fX0*N))] = 1.0
        X = get_random_state(X,fX0)
    else:
        X = get_ordered_state(X,fX0)
    return X


_run_transient_output_type = types.Tuple((types.int64[:],types.int64,types.float64[:],types.int64))
_run_transient_input_type  = (_type_state_iter,types.int64,types.float64,types.int64[:],types.int64,types.float64,types.boolean,types.boolean,types.int64[:,:],types.int64[:])
@njit(_run_transient_output_type(*_run_transient_input_type))
def Run_transient_sequential(state_iter,tTrans_eff,alpha,X,M,fX0,is_aval_sim,is_meanfield,neigh,K):
    N                   = len(X)
    N_fl                = float(len(X))
    sum_X               = sum(X)
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,fX0)
    for t in range(1,tTrans_eff):
        continue_time_loop, X, sum_X = check_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count)
        if not continue_time_loop:
            break
        i                   = numpy.random.randint(0,N) # selecting update site
        Xa                  = X[i]
        if is_meanfield:
            n_act_neigh     = float(sum_X-X[i])/(N_fl-1.0)
        else:
            n_act_neigh     = sum(X[neigh[i,:K[i]]])/float(K[i]) #sum(X[neigh[i]])/float(len(neigh[i]))
        X[i]                = state_iter(X[i],n_act_neigh,alpha) # updating site i
        sum_X              += X[i] - Xa # +1 if activated i; -1 if deactivated i
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,float(sum_X) / N_fl)
    return X,sum_X,rho_memory,cs_count

@njit(_run_transient_output_type(*_run_transient_input_type))
def Run_transient_parallel(state_iter,tTrans,alpha,X,M,fX0,is_aval_sim,is_meanfield,neigh,K):
    N                   = len(X)
    N_fl                = float(len(X))
    sum_X               = sum(X)
    rho_prev            = float(sum_X) / N_fl
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,fX0)
    for t in range(1,tTrans):
        # updates sum_X and X as needed if the network activity must be restarted
        continue_time_loop, X, sum_X = check_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count)
        if not continue_time_loop:
            break
        if is_meanfield:
            n_act_neigh = rho_prev
        else:
            X_prev      = X.copy()
        sum_X           = 0
        for i in range(N):
            if not is_meanfield:
                #n_act_neigh = sum(X_prev[neigh[i]])/float(len(neigh[i]))
                n_act_neigh = sum(X_prev[neigh[i,:K[i]]])/float(K[i])
            X[i]   = state_iter(X[i],n_act_neigh,alpha)
            sum_X += X[i]
        rho_prev            = float(sum_X) / N_fl
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,rho_prev) 
    return X,sum_X,rho_memory,cs_count

_type_simulation_result  = types.Tuple((types.float64[:],types.float64[:],_type_X_data))
_type_mfsimulation_input = (types.int64,types.int64,types.float64,types.boolean,types.float64,types.int64,types.int64,types.float64,types.int64,types.int64,types.int64,types.int64,types.boolean,types.boolean,types.string)
@njit(_type_simulation_result(*_type_mfsimulation_input))
def Run_MF_parallel(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,dtsample,M,iterdynamics,sim,saveSites,writeOnRun,spkFileName):
    # all sites update in the same time step -- matches the GL model
    X                   = get_IC(X0, fX0, X0Rand, N)     
    is_aval_sim         = sim == Options_sim.AVAL
    state_iter          = get_site_state_iterator(iterdynamics)
    alpha               = get_site_state_iterator_alpha(iterdynamics,l)
    N_fl                = float(N)
    dtsample            = int(dtsample-1) if dtsample > 1 else 1

    X,sum_X,rho_memory,cs_count  = Run_transient_parallel(state_iter,tTrans,alpha,X,M,fX0,is_aval_sim,True,numpy.zeros((0,2),dtype=numpy.int64),numpy.zeros((0,),dtype=numpy.int64))
    rho_prev                     = float(sum_X) / N_fl

    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun, False)
    X_data                       = save_initial_network_state(X, 0.0, saveSites, writeOnRun)


    n_data              = 1 + (tTotal - tTrans - 1) // dtsample    
    rho                 = numpy.zeros(n_data, dtype=numpy.float64) # numpy.zeros(tTotal-tTrans, dtype=numpy.float64)
    time                = numpy.zeros(n_data, dtype=numpy.float64) # numpy.zeros(tTotal-tTrans, dtype=numpy.float64)
    trec                = 0
    rho[trec]           = rho_prev
    time[trec]          = 0.0
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,rho_prev)
    for t in range(1,tTotal-tTrans):
        continue_time_loop, X, sum_X = check_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count)
        if not continue_time_loop:
            break
        rho_prev = float(sum_X) / N_fl
        sum_X    = 0
        if t%dtsample == 0: # we only save spikes for time steps multiples of dtsample
            trec      += 1
            for i in range(N):
                X[i]   = state_iter(X[i],rho_prev,alpha)
                sum_X += X[i]
                X_data = save_spk_time(X_data, t, i, X[i]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
                _      = write_spk_time(X_data, t, i, X[i])               # this function can just be a dummy placeholder depending on saveSites and writeOnRun
            rho[trec]  = float(sum_X) / N_fl
            time[trec] = float(t)
        else:
            for i in range(N):
                X[i]   = state_iter(X[i],rho_prev,alpha)
                sum_X += X[i]
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,rho[t])
    close_file(spk_file,spkFileName,saveSites and writeOnRun)
    return rho, time, X_data

@njit(_type_simulation_result(*_type_mfsimulation_input))
def Run_MF_sequential(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,dtsample,M,iterdynamics,sim,saveSites,writeOnRun,spkFileName):
    # only 1 site is attempted update at each time step
    X                   = get_IC(X0, fX0, X0Rand, N)
    is_aval_sim         = sim == Options_sim.AVAL
    state_iter          = get_site_state_iterator(iterdynamics)
    alpha               = get_site_state_iterator_alpha(iterdynamics,l)
    N_fl                = float(N)
    tTrans_eff          = int(numpy.round(tTrans / dt))
    tTotal_eff          = int(numpy.round(tTotal / dt))
    dtsample            = int(dtsample-1) if dtsample>1 else 1
    dtsample_is_1       = dtsample == 1
    n_neigh             = N_fl - 1.0

    X,sum_X,rho_memory,cs_count  = Run_transient_sequential(state_iter,tTrans_eff,alpha,X,M,fX0,is_aval_sim,True,numpy.zeros((0,2),dtype=numpy.int64),numpy.zeros((0,),dtype=numpy.int64))
    
    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_data                       = save_initial_network_state(X, 0.0, saveSites, writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun, dtsample_is_1)
    dump_data                    = (saveSites or writeOnRun) and (dtsample > 1)

    n_data              = 1 + (tTotal_eff - tTrans_eff - 1) // dtsample
    rho                 = numpy.full(n_data, numpy.nan, dtype=numpy.float64)
    time                = numpy.full(n_data, numpy.nan, dtype=numpy.float64)
    sum_X               = sum(X)
    rho[0]              = float(sum_X) / N_fl
    time[0]             = 0.0
    trec                = 0
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,rho[0])
    for t in range(1,tTotal_eff-tTrans_eff):
        # deciding whether to reseed activity...
        # this allows us to record the visit to the absorbing state to split avalanches in the future analysis
        continue_time_loop, X, sum_X = check_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count)
        if not continue_time_loop:
            break
        i      = numpy.random.randint(0,N) # selecting update site
        Xa     = X[i]
        X[i]   = state_iter(X[i],float(sum_X-X[i])/n_neigh,alpha) # updating site i
        sum_X += X[i] - Xa  # +1 if activated i; -1 if deactivated i
        if t%dtsample == 0: # we only save spikes for time steps multiples of dtsample
            trec      += 1
            X_data     = dump_spike_data_sequential_update(dump_data,save_spk_time,write_spk_time,X_data,dtsample_is_1,t,dt,i,X,Xa)
            rho[trec]  = float(sum_X) / N_fl
            time[trec] = t*dt
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,rho[t])
    close_file(spk_file,spkFileName,saveSites and writeOnRun)
    return rho, time, X_data

type_netsimulation_input = (types.int64,types.int64,types.float64,types.boolean,types.float64,types.int64,types.int64,types.float64,types.int64,types.int64,types.int64,types.int64 ,types.int64,types.boolean,types.boolean,types.string)
@njit(_type_simulation_result(*type_netsimulation_input))
def Run_Graph_parallel(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,dtsample,M,graph,iterdynamics,sim,saveSites,writeOnRun,spkFileName):
    X                   = get_IC(X0, fX0, X0Rand, N)
    neigh,K             = get_graph_neighbors(graph,N) #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    is_aval_sim         = sim == Options_sim.AVAL
    state_iter          = get_site_state_iterator(iterdynamics)
    alpha               = get_site_state_iterator_alpha(iterdynamics,l)
    N_fl                = float(N)
    dtsample            = int(dtsample-1) if dtsample > 1 else 1

    X,sum_X,rho_memory,cs_count  = Run_transient_parallel(state_iter,tTrans,alpha,X,M,fX0,is_aval_sim,False,neigh,K)

    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun, False)
    X_data                       = save_initial_network_state(X, 0.0, saveSites, writeOnRun)

    n_data              = 1 + (tTotal - tTrans - 1) // dtsample    
    rho                 = numpy.zeros(n_data, dtype=numpy.float64) # numpy.zeros(tTotal-tTrans, dtype=numpy.float64)
    time                = numpy.zeros(n_data, dtype=numpy.float64) # numpy.zeros(tTotal-tTrans, dtype=numpy.float64)
    trec                = 0
    rho[trec]           = float(sum_X) / N_fl
    time[trec]          = 0.0
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,rho[0])
    for t in range(1,tTotal-tTrans):
        continue_time_loop, X, sum_X = check_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count)
        if not continue_time_loop:
            break
        X_prev = X.copy()
        sum_X  = 0
        if t%dtsample == 0: # we only save spikes for time steps multiples of dtsample
            trec      += 1
            for i in range(N):
                #X[i]   = state_iter(X[i],sum(X_prev[neigh[i]])/float(len(neigh[i])),alpha)
                X[i]   = state_iter(X[i],sum(X_prev[neigh[i,:K[i]]])/float(K[i]),alpha)
                sum_X += X[i]
                X_data = save_spk_time(X_data, t, i, X[i]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
                _      = write_spk_time(X_data, t, i, X[i])               # this function can just be a dummy placeholder depending on saveSites and writeOnRun
            rho[trec]  = float(sum_X) / N_fl
            time[trec] = float(t)
        else:
            for i in range(N):
                #X[i]   = state_iter(X[i],sum(X_prev[neigh[i]])/float(len(neigh[i])),alpha)
                X[i]   = state_iter(X[i],sum(X_prev[neigh[i,:K[i]]])/float(K[i]),alpha)
                sum_X += X[i]
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,rho[t])
    close_file(spk_file,spkFileName,saveSites and writeOnRun)
    return rho, time, X_data

@njit(_type_simulation_result(*type_netsimulation_input))
def Run_Graph_sequential(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,dtsample,M,graph,iterdynamics,sim,saveSites,writeOnRun,spkFileName):
    X                   = get_IC(X0,fX0,X0Rand,N)
    neigh,K             = get_graph_neighbors(graph,N) #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    is_aval_sim         = sim == Options_sim.AVAL
    state_iter          = get_site_state_iterator(iterdynamics)
    alpha               = get_site_state_iterator_alpha(iterdynamics,l)
    N_fl                = float(N)
    tTrans_eff          = int(numpy.round(tTrans / dt)) # this expands time only if dt=1/N (i.e. expandtime==True)
    tTotal_eff          = int(numpy.round(tTotal / dt)) # this expands time only if dt=1/N (i.e. expandtime==True)
    dtsample            = int(dtsample-1) if dtsample>1 else 1
    dtsample_is_1       = dtsample == 1

    X,sum_X,rho_memory,cs_count  = Run_transient_sequential(state_iter,tTrans_eff,alpha,X,M,fX0,is_aval_sim,False,neigh,K)
    
    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun, dtsample_is_1)
    X_data                       = save_initial_network_state(X, 0.0, saveSites, writeOnRun)
    dump_data                    = (saveSites or writeOnRun) and (dtsample > 1)

    n_data              = 1 + (tTotal_eff - tTrans_eff - 1) // dtsample
    rho                 = numpy.full(n_data, numpy.nan, dtype=numpy.float64)
    time                = numpy.full(n_data, numpy.nan, dtype=numpy.float64)
    sum_X               = sum(X)
    rho[0]              = float(sum_X) / N_fl
    time[0]             = 0.0
    trec                = 0
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,rho[0])
    for t in range(1,tTotal_eff-tTrans_eff):
        # deciding whether to reseed activity...
        # this allows us to record the visit to the absorbing state to split avalanches in the future analysis
        continue_time_loop, X, sum_X = check_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count)
        if not continue_time_loop:
            break
        i      = numpy.random.randint(0,N) # selecting update site
        Xa     = X[i]
        #X[i]   = state_iter(X[i],sum(X[neigh[i]])/float(len(neigh[i])),alpha) # updating site i
        X[i]   = state_iter(X[i],sum(X[neigh[i,:K[i]]])/float(K[i]),alpha) # updating site i
        sum_X += X[i] - Xa  # +1 if activated i; -1 if deactivated i
        if t%dtsample == 0: # we only save spikes for time steps multiples of dtsample
            trec      += 1
            X_data     = dump_spike_data_sequential_update(dump_data,save_spk_time,write_spk_time,X_data,dtsample_is_1,t,dt,i,X,Xa)
            rho[trec]  = float(sum_X) / N_fl
            time[trec] = t*dt
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,float(sum_X) / N_fl)
    close_file(spk_file,spkFileName,saveSites and writeOnRun)
    return rho, time, X_data
