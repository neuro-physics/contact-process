import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning
warnings.simplefilter('ignore', NumbaExperimentalFeatureWarning)

import os
import sys
import time as ostime
import numpy
import argparse
import datetime
import contextlib
import modules.io as io
import modules.cpsim_io_options as ioopt
import modules.cpsim_numba as cp

def main(): 

    # for debug
    #sys.argv = 'python contact_process.py -l 3.4 -N 10000 -tTotal 1000 -graph ring -X0 1 -outputFile cp_ring.mat'.split()[1:]
    #sys.argv  = 'python contact_process.py -l 3.0 -N 1000 -tTotal 10000 -graph ring -X0 1 -outputFile cp_ring_l3.0_aval_N1000_t10000.mat -saveSites -writeOnRun'.split()[1:]
    #sys.argv  = 'python contact_process.py -l 3.000000 -N 2560 -tTrans 50000 -tTotal 80000 -graph ring -X0 1 -fX0 0 -sim aval -outputFile test_sim/N2560/corr_ring/cp_ring_l3.0_par_aval_N2560_t80000_fX00.mat      -saveSites -writeOnRun -update parallel -mergespkfile'.split()[1:]
    #sys.argv = 'python contact_process.py -l 3.000000 -N 1000 -tTrans 1000 -tTotal 1100 -graph ring -X0 1 -fX0 0 -sim aval -outputFile debug/cp_ring_l3.0_par_aval_N10000_t50000_fX00.mat      -saveSites -writeOnRun -update parallel -mergespkfile'.split()[1:]
    #sys.argv = 'python contact_process.py -l 0.7 -N 2560 -tTrans 50000 -tTotal 80000 -graph alltoall -X0 1 -fX0 0 -sim aval -outputFile test_sim/N2560/corr_mf/cp_mf_l0.7_par_aval_N2560_t80000_fX00.mat -saveSites -writeOnRun -update parallel -mergespkfile'.split()[1:]
    parser = argparse.ArgumentParser(description='Contact process in 1+1 dimensions or mean-field all-to-all graph\n\n(l_c=3.297848 for ring; l_c=1 for mean-field)',formatter_class=argparse.RawTextHelpFormatter)
    parser = io.add_simulation_parameters(parser)

    args              = io.parse_input_args(parser.parse_args())

    # validating inputs to the simulation
    #args.sim          = ioopt.str_to_Options_sim(args.sim)
    #args.graph        = ioopt.str_to_Options_graph(args.graph)
    #args.update       = ioopt.str_to_Options_update(args.update)
    #args.iterdynamics = ioopt.str_to_Options_iterdynamics(args.iterdynamics)
    args.docstring    = io.get_help_string(parser)
    args.X0Rand       = not args.noX0Rand
    args.LyLx         = numpy.array(cp._largest_close_factors(args.N))
    args.dt           = cp.Get_Simulation_Timescale(args) # dt = 1/N if (not parallel update) and (expandtime); otherwise, dt=1
    args.expandtime   = args.expandtime and not cp.is_parallel_update(args.update) # expandtime==True only if it is True AND it isn't a parallel update simulation
    args.outputFile   = io.get_new_file_name(io.get_output_filename(args.outputFile))
    args.spkFileName  = args.outputFile.replace('.mat','_spk.txt') if args.writeOnRun else ''

    #print('f=',args.outputFile)
    #print(len(args.outputFile))
    #
    #exit()

    output_dir       = os.path.dirname(args.outputFile)
    os.makedirs(output_dir, exist_ok=True)

    print('* Simulation parameters:')
    print(args)

    print("* Running simulation... Total time steps = %d" % (int(numpy.round(args.tTotal / args.dt))))
    simulation_func = cp.Get_Simulation_Func(args)
    sim_time_start  = ostime.monotonic()


    with (open(args.spkFileName, 'w') if (args.writeOnRun and args.saveSites) else contextlib.nullcontext(sys.stdout)) as spk_file:
        if args.writeOnRun and args.saveSites:
            print('  ... writing file ',args.spkFileName,' during simulation')
        with contextlib.redirect_stdout(spk_file):
            sim_params_dict   = io.keep_keys(dict(**args),io.get_func_param(simulation_func))
            rho, time, X_data = simulation_func(**sim_params_dict)
    
    ##### FOR DEBUG ... disable redirection of stdout
    #rho, X_data     = simulation_func(**io.keep_keys(dict(**args),io.get_func_param(simulation_func)))


    sim_time_end    = ostime.monotonic()
    print("* End of simulation... Total time: {}".format(datetime.timedelta(seconds=sim_time_end - sim_time_start)))

    print('* Writing ... %s'%args.outputFile)
    io.save_simulation_file(sys.argv, args, rho, time, X_data)
    del rho, time, X_data # releasing memory to avoid memory error when merging files
    if args.saveSites and args.writeOnRun and args.mergespkfile:
        io.merge_simulation_spike_files(args.outputFile, args.spkFileName, remove_spk_file=False, verbose=True)

    print('done')
    print(' ')



if __name__ == '__main__':
    main()
