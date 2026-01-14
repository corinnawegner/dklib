import argparse

def add_steppable_params(base_parser:argparse.ArgumentParser, steppable_parameters, flaggable_parameters, parameter_types:dict ):
    """
        Adds steppable and flaggable parameters to a base argument parser.
        Arguments:
            base_parser (argparse.ArgumentParser):
                The base parser to which parameters will be added.
            steppable_parameters (list):
                List of parameter names that can take a range of values.
            flaggable_parameters (list):
                List of parameter names that are boolean flags.
            parameter_types (dict):
                Dictionary mapping parameter names to their types.
        Example: 
            steppable_parameters = ['L','h', 'mask_density','replicate','lr','Ntrain','m','batch_size','weight_decay','representation_alignment_loss_weight','teacher_lag_final']
            flaggable_parameters = ['conv_after_skip','masked_weighting','MF_weight_rescaling']
            parameter_types = {'L':int,'h':int,'mask_density':float,'replicate':int,'lr':float, 'Ntrain':int,'m':int,'batch_size':int,'weight_decay':float,'representation_alignment_loss_weight':float,'teacher_lag_final':float,}
    """
    for param in steppable_parameters:
        base_parser.add_argument(f'--{param}_min',type=parameter_types[param],default=None,help=f'Minimum value for {param}.')
        base_parser.add_argument(f'--{param}_max',type=parameter_types[param],default=None,help=f'Maximum value for {param}.')
        base_parser.add_argument(f'--{param}_Nstep',type=int,default=None,help=f'Number of steps for {param}.')
        base_parser.add_argument(f'--{param}_geom',action='store_true',help=f'Vary the steps of {param} geometrically.')

    for param in flaggable_parameters:
        parameter_types[param] = bool
        base_parser.add_argument(f'--{param}_vary',action='store_true',help=f'If set, will test {param} with both True and False.')
    return base_parser

def add_slurm_parameters(base_parser:argparse.ArgumentParser):
    """
    Add SLURM-related command-line arguments to an existing ArgumentParser.
    Parameters
    ----------
    base_parser : argparse.ArgumentParser
        The argument parser to augment with SLURM and job-splitting options.
    Returns
    -------
    argparse.ArgumentParser
        The same parser instance with the following additional arguments:
          --N_jobs : int, default=1
              Number of parallel jobs to split the workload across.
          --job_index : int, default=0
              Index of the current job in the parallel job set.
          --min_job_index : int, default=0
              Starting index for SLURM job arrays when the array index is non-zero.
          --start_permutation_index : int, default=0
              Index of the first permutation to run (for partial or restarted sweeps).
          --end_permutation_index : int, default=None
              Index of the last permutation to run; if None, runs through the final permutation.
          --debug_params : bool (flag)
              If set, prints the full parameter grid for debugging and exits immediately.
          --output_filename_prefix : str, default=None
              Prefix to prepend to output filenames when saving results.
    """
    
    #wall time related parameters: 
    # Add wall_time_limit only if it's not already defined
    if '--wall_time_limit' not in base_parser._option_string_actions:
        base_parser.add_argument(
            '--wall_time_limit',
            type=int,
            default=300,
            help='Maximum wall time limit for the job in seconds. Default is 5 minutes.'
        )

    #splitting the job across multiple runs.
    base_parser.add_argument('--N_jobs',type=int,default=1,help='Number of parallel jobs')
    base_parser.add_argument('--job_index',type=int,default=0,help='Index of the current job')
    base_parser.add_argument('--min_job_index',type=int,default=0,help='Index of the first worker -- relevant if we are running a slurm job array, where we start at a nonzero worker index.')
    #sometimes we only want to run a subset of the indices, say to restart a failed or incomplete job.
    base_parser.add_argument('--start_permutation_index',type=int,default=0,help='Index of the first permutation to run.')
    base_parser.add_argument('--end_permutation_index',type=int,default=None,help='Index of the last permutation to run. If None, defaults to the last permutation.')
    base_parser.add_argument('--skip_existing_permutations',action='store_true',help='If set, will skip permutations for which output files already exist.')
    base_parser.add_argument('--existing_permutations_output_prefix',type=str,default=None,help='If set, will look for existing output files with this prefix to determine which permutations to skip, instead of using the output_filename_prefix.')
    #debugging the parameters to sweep through.
    base_parser.add_argument('--debug_params',action='store_true',help='If set, will print out the parameter grid and exit.')
    #file output: 
    base_parser.add_argument('--output_filename_prefix',type=str,default=None,help='Output file prefix to save the results.')
    #Add an argument that allows us to associate tuples of parameters with each other. 
    #The tuples need to be of the form (param1,param2,...), where each param is a parameter name.
    #We also need to indicate whether the parameters that are swept together are done so in the forward or backwards direction.
    # base_parser.add_argument('--sweep_tuples',type=str,default=None,help='A string representation of a list of tuples of parameters to sweep together. Each tuple should be of the form (param1,param2,...), where each param is a parameter name. The tuples should be separated by commas. If this argument is not provided, all varied parameters will be treated as independent sweep directions. No parameter should appear in more than one tuple.')
    # base_parser.add_argument('--sweep_tuples_forward',type=str,default=None,help='A string represention of a list of tuples of parameters, each of which should be of the form (True,False,...). If set, each True/False indicates whether that parameter should be swept in the forward or reverse direction, and directions for all tuples must be specified. If this argument is not provided, all will be treated as forward.')
    return base_parser 


def build_parameter_grid(args, steppable_parameters, flaggable_parameters, parameter_types):
    
    #=== Setting up the parameter grid
    import numpy as np 
    param_grid = {}
    for param in steppable_parameters:
        if(getattr(args,f'{param}_min') is not None):
            if((getattr(args,f'{param}_max') is None or 
               getattr(args,f'{param}_Nstep') is None)  and parameter_types[param] is int ):
                if(getattr(args,f'{param}_max') is None and 
                getattr(args,f'{param}_Nstep') is None):
                    raise ValueError(f'Parameter {param} must have either a max or Nstep defined.')
                elif (getattr(args,f'{param}_max') is None ):
                    print('WARNING: Parameter %s_max is missing, setting it equal to _min+_Nstep-1'%(param))
                    setattr(args,f'{param}_max',getattr(args,f'{param}_min')+getattr(args,f'{param}_Nstep')-1)
                elif (getattr(args,f'{param}_Nstep') is None and parameter_types[param] is int):
                    print('WARNING: Parameter %s_Nstep is missing, setting it equal to _max - _min +1'%(param))
                    setattr(args,f'{param}_Nstep', getattr(args,f'{param}_max')-getattr(args,f'{param}_min')+1)

            assert(getattr(args,f'{param}_max') is not None)
            assert(getattr(args,f'{param}_Nstep') is not None)
            if(getattr(args,f'{param}_geom')):
                spacer_func = np.geomspace
            else: 
                spacer_func = np.linspace
            param_grid[param] = spacer_func(getattr(args,f'{param}_min'),getattr(args,f'{param}_max'),getattr(args,f'{param}_Nstep'),dtype=parameter_types[param])
    for param in flaggable_parameters: 
        if(getattr(args,f'{param}_vary')):
            param_grid[param] = np.array([True,False],dtype=bool)

    import itertools
    permutations = list(itertools.product( *param_grid.values() ))
    N_permutations = len(permutations)
    print('number of runnable permutations: ',N_permutations)

    #selecting which permutations this iteration will run: 
    if(args.end_permutation_index is None):
        args.end_permutation_index = N_permutations
    start_indices = np.linspace(args.start_permutation_index,args.end_permutation_index,args.N_jobs+1).astype(int)
    inds_to_run = np.arange(start_indices[args.job_index-args.min_job_index],start_indices[args.job_index+1-args.min_job_index])
    print('this thread will run indices: ',inds_to_run)
    permutations_to_run = permutations[start_indices[args.job_index-args.min_job_index]:start_indices[args.job_index+1-args.min_job_index]]
    if(args.debug_params):
        print(param_grid)
        print(permutations)
        print('this thread will run permuations: ',permutations_to_run)
    return permutations_to_run, inds_to_run, param_grid

def run_sweep(base_parser, steppable_parameters, flaggable_parameters, parameter_types, experiment_function, return_outputs=False, WALL_TIME_BUFFER=60):
    """
    Run a parameter sweep over a grid of hyperparameters with SLURM support.
    This function augments the provided argument parser with steppable and SLURM-specific
    options, constructs all combinations of the specified parameters, and invokes the
    given experiment function for each permutation. It manages the remaining wall time
    to prevent job overruns, saves individual outputs to pickle files (if an output
    filename prefix is provided), and optionally returns all outputs.
    Args:
        base_parser (argparse.ArgumentParser):
            The initial argument parser to which sweep and SLURM parameters will be added.
        steppable_parameters (Mapping[str, Iterable]):
            A mapping from parameter names to sequences of values that define the sweep grid.
        flaggable_parameters (Iterable[str]):
            A list of parameter names that should be treated as boolean flags.
        parameter_types (Mapping[str, Callable[[Any], Any]]):
            A mapping from parameter names to type‐conversion functions for parsed values.
        experiment_function (Callable[[argparse.Namespace], Any]):
            A function that executes the experiment. It receives the parsed args namespace
            (with parameters set for the current permutation) and returns an output object.
        return_outputs (bool, optional):
            If True, collect and return a list of outputs from each experiment run.
            Defaults to False.
        WALL_TIME_BUFFER (int, optional):
            A safety margin (in seconds) subtracted from the SLURM wall time limit to
            avoid overruns. Defaults to 60 seconds.
    Returns:
        List[Any] or None:
            If return_outputs is True, returns a list of outputs from each call to
            experiment_function. Otherwise, returns None.
    Side Effects:
        - Parses command-line arguments.
        - Iterates through the parameter grid, setting attributes on the args namespace.
        - Checks and updates the remaining wall time before each run.
        - Prints progress and timing information to stdout.
        - Saves each experiment's output to a pickle file named
          '{output_filename_prefix}permutation_{index}.pkl' if a prefix is provided.
    """
    
    #adding additional sweep parameters to the base parser
    base_parser = add_steppable_params(base_parser, steppable_parameters, flaggable_parameters, parameter_types)
    base_parser = add_slurm_parameters(base_parser)
    args = base_parser.parse_args()
    #figuring out which parameters we need to run: 
    permutations_to_run, inds_to_run, param_grid = build_parameter_grid(args, steppable_parameters, flaggable_parameters, parameter_types)
    
    import time
    wtime_start = time.time()
    
    #Setting the wall time limit
    initial_max_walltime = args.wall_time_limit

    def fname(permutation_ind,prefix = args.output_filename_prefix):
        return prefix+f'permutation_{permutation_ind}.pkl'
    
    experiment_outputs = [] 
    for i,permutation_ind in enumerate(inds_to_run):
        permutation = permutations_to_run[i]
        print('============')
        print('starting permutation: %d. '%permutation_ind,'Thread job progress: ',i+1,'/',len(permutations_to_run), ' Parameters: ')
        for param, value in zip(param_grid.keys(),permutation):
            print(param+': ',parameter_types[param](value.item()))
            setattr(args,param,parameter_types[param](value.item()))
        print('---',flush=True)
        if(not args.debug_params):
            args.wall_time_limit = initial_max_walltime - (time.time()-wtime_start) - WALL_TIME_BUFFER
            print('remaining wall time limit: %.1f'%args.wall_time_limit,'seconds')
            if(args.wall_time_limit < 0.0):
                print('Wall time limit exceeded. Exiting.')
                break
            if(args.skip_existing_permutations and args.output_filename_prefix is not None):
                import os
                prefix = args.existing_permutations_output_prefix if args.existing_permutations_output_prefix is not None else args.output_filename_prefix
                if(os.path.exists(fname(permutation_ind,prefix=prefix))):
                    print('Output file %s already exists. Skipping permutation %d.'%(fname(permutation_ind,prefix=prefix),permutation_ind))
                    continue
            wtime_experiment_start = time.time()
            experiment_output = experiment_function(args)
            if(args.output_filename_prefix is not None):
                import pickle
                output = experiment_output 
                with open(fname(permutation_ind),'wb') as f:
                    pickle.dump(output,f)
            if(return_outputs):
                experiment_outputs.append(experiment_output)
            print('finished permutation: %d. '%permutation_ind,'Thread job progress: ',i+1,'/',len(permutations_to_run), ' Total time elapsed: %.1f seconds, experiment took: %.1f seconds'%(time.time()-wtime_start, time.time()-wtime_experiment_start),flush=True)
    if(return_outputs):
        return experiment_outputs