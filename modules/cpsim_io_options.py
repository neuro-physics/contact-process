import sys
from enum import IntEnum

class Options_graph(IntEnum):
    ALLTOALL       = 0
    RING           = 1
    RINGFREE       = 2
    SQUAREPERIODIC = 3
    SQUAREFREE     = 4

class Options_iterdynamics(IntEnum):
    TOME_OLIVEIRA    = 0
    MARRO_DICKMAN    = 1

class Options_sim(IntEnum):
    TIMEEVO = 0
    AVAL    = 1

class Options_update(IntEnum):
    PARALLEL   = 0
    SEQUENTIAL = 1


def _create_option_choice_to_enum_map():
    """
    creates a map between a simulation input parameter
    and the corresponding enum above.

    this is useful to standardize the conversion between an input parameter to the simulation,
    and the appropriate option given in the corresponding enum.

    For example,
    > _OPTION_ENUM_MAP_ = create_option_choice_to_enum_map()
    then,
        _OPTION_ENUM_MAP_ = {
            'parScale'       : Options_parScale,
            'inhibitionModel': Options_inhibitionModel,
            'weightDynType'  : Options_weightDynType,
            'netType'        : Options_netType,
            'simType'        : Options_simType
        }
    """
    option_enum_map = {}
    current_module = sys.modules[__name__]
    for name in dir(current_module):
        obj = getattr(current_module, name)
        if isinstance(obj, type) and issubclass(obj, IntEnum):
            if 'Options_' in name:
                option_enum_map[name.split('_')[1]] = obj
    return option_enum_map


#_OPTION_ENUM_MAP_ = dict()
#def create_option_choice_to_enum_map():
#    _OPTION_ENUM_MAP_ = {
#        'update'       : Options_update,
#        'sim'          : Options_sim,
#        'graph'        : Options_graph,
#        'iterdynamics' : Options_iterdynamics
#    }
_OPTION_ENUM_MAP_ = _create_option_choice_to_enum_map()


def has_options_enum(option):
    return option in _OPTION_ENUM_MAP_

def convert_option_choice_to_enum(option, choice):
    """
    converts a string choice of the option simulation input parameter
    to its corresponding enum value.

    option -> str, an option parameter from simulation parameters (e.g., 'parScale' or 'netType')
    choice -> str, the choice (enum label) for the corresponding option (e.g., if option=='parScale', choices are 'log' or 'linear')

    returns
        v -> Options_OPTION.CHOICE the corresponding enum value
        where OPTION is the suffix of one of Options_ enum above,
        and CHOICE is the enum label
    
    example
    > v = convert_option_choice_to_enum('parScale','log') # v == Options_parScale.log
    """
    if len(_OPTION_ENUM_MAP_) == 0:
        raise ValueError('You must initialize the conversion function by calling: _OPTION_ENUM_MAP_ = create_option_choice_to_enum_map()')
    if option in _OPTION_ENUM_MAP_:
        try:
            if option == 'graph':
                return str_to_Options_graph(choice)
            elif option == 'sim':
                return str_to_Options_sim(choice)
            elif option == 'update':
                return str_to_Options_update(choice)
            elif option == 'iterdynamics':
                return str_to_Options_iterdynamics(choice)
            return _OPTION_ENUM_MAP_[option][choice]
        except KeyError as e:
            raise ValueError(f"Unknown choice for {option}: {choice}") from e
    else:
        raise ValueError(f"Unknown option: {option}")

def convert_intvalue_to_option_enum(option, value):
    """
    converts an int choice of the option simulation input parameter
    to its corresponding string value.

    option -> str, an option parameter from simulation parameters (e.g., 'parScale' or 'netType')
    choice -> int, the choice (enum value) for the corresponding option (e.g., if option=='parScale', choices are 0 for 'log' or 1 for 'linear')

    returns
        v -> Options_OPTION.CHOICE the corresponding enum value
        where OPTION is the suffix of one of Options_ enum above,
        and CHOICE is the label that corresponds to the enum value

    example
    > v = convert_option_choice_to_enum('parScale',0) # v == Options_parScale.log
    """
    if len(_OPTION_ENUM_MAP_) == 0:
        raise ValueError('You must initialize the conversion function by calling: _OPTION_ENUM_MAP_ = create_option_choice_to_enum_map()')
    if option in _OPTION_ENUM_MAP_:
        try:
            return _OPTION_ENUM_MAP_[option](value)
        except ValueError as e:
            raise ValueError(f"Unknown value for {option}: {value}") from e
    else:
        raise ValueError(f"Unknown option: {option}")

def convert_to_enum(option,choice_str_or_int):
    if type(choice_str_or_int) is str:
        return convert_option_choice_to_enum(option,choice_str_or_int)
    elif type(choice_str_or_int) is int:
        return convert_intvalue_to_option_enum(option,choice_str_or_int)
    else:
        raise TypeError(f'Unknown type for choice_str_or_int')

def str_to_Options_graph(graph_str):
    graph_str = graph_str.lower()
    if (graph_str == 'alltoall') or (graph_str == 'mf'):
        return Options_graph.ALLTOALL
    elif (graph_str == 'ring'):
        return Options_graph.RING
    elif (graph_str == 'ringfree'):
        return Options_graph.RINGFREE
    elif (graph_str == 'squareperiodic'):
        return Options_graph.SQUAREPERIODIC
    elif (graph_str == 'squarefree'):
        return Options_graph.SQUAREFREE
    else:
        raise ValueError(f'Unknown graph type: {graph_str}')

def str_to_Options_sim(sim_str):
    sim_str = sim_str.lower()
    if (sim_str == 'timeevo'):
        return Options_sim.TIMEEVO
    elif (sim_str == 'aval'):
        return Options_sim.AVAL
    else:
        raise ValueError(f'Unknown simulation type: {sim_str}')

def str_to_Options_update(updt_str):
    updt_str = updt_str.lower()
    if (updt_str == 'parallel') or (updt_str == 'par'):
        return Options_update.PARALLEL
    elif (updt_str == 'sequential') or (updt_str == 'seq'):
        return Options_update.SEQUENTIAL
    else:
        raise ValueError(f'Unknown update type: {updt_str}')

def str_to_Options_iterdynamics(itype_str):
    itype_str = itype_str.lower()
    if (itype_str == 'tome_oliveira') or (itype_str == 'to') or (itype_str == 'tomeoliveira'):
        return Options_iterdynamics.TOME_OLIVEIRA
    elif (itype_str == 'marro_dickman') or (itype_str == 'md') or (itype_str == 'marrodickman'):
        return Options_iterdynamics.MARRO_DICKMAN
    else:
        raise ValueError(f'Unknown state iterator type: {itype_str}')

def is_parallel_update(update:Options_update):
    return update == Options_update.PARALLEL
