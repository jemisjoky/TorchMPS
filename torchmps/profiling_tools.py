from time import perf_counter
from functools import partial

import torch
from torch.profiler import profile, record_function

from torchmps import ProbMPS
from torchmps.embeddings import (
    FixedEmbedding,
    unit_interval,
    trig_embed,
    legendre_embed,
)

# Model and data parameters
MNIST_SIZE = 28 ** 2
BATCH_SIZE = 128
NUM_BATCH = 1
INPUT_DIM = 5
BOND_DIM = 100
COMPLEX = False
JITTED = False
TRACE_JIT = True  # Whether to use jit.trace (True) or jit.script (False)
# EMBED_FUN = trig_embed  # Set to None for discrete modeling
EMBED_FUN = None  # Set to None for discrete modeling
DTYPE = torch.float32
EVAL_TYPE = "slim"  # Options: "slim", "default", "parallel"

# Profiling parameters
SAVE_TRACE = False  # Whether to save trace of program
WITH_STACK = False  # Whether to record call stack
STACK_DEPTH = 1  # Number of call frames to display
FAKE_PROFILE = False  # Whether to run code without profiling
PROFILE_MEMORY = True
INCLUDE_BACKWARD = False
SORT_ATTR = "cpu_time_total"


# Function for setting derived globals
def set_derived_globals():
    global SLIM_EVAL, PARALLEL_EVAL
    assert EVAL_TYPE in ["slim", "default", "parallel"]
    SLIM_EVAL = EVAL_TYPE == "slim"
    PARALLEL_EVAL = EVAL_TYPE == "parallel"


set_derived_globals()


def restore_config(foo):
    """
    Decorator for returning global config variables to their original values
    """

    def wrapped_foo(*args, **kwargs):
        # Save original set of configuration values
        glob_dict = globals()
        config_keys = [k for k in glob_dict.keys() if k.upper() == k]
        orig_config = {k: glob_dict[k] for k in config_keys}

        # Call foo, which might change these config variables
        out = foo(*args, **kwargs)

        # Return config variables to their original values
        globals().update(orig_config)

        return out

    return wrapped_foo


def init_data():
    """
    Generate batches of discrete or continuous fake data of same size as MNIST
    """
    shape = (NUM_BATCH, BATCH_SIZE, MNIST_SIZE)

    if EMBED_FUN is None:
        return torch.randint(INPUT_DIM, shape)
    else:
        return torch.rand(shape)


def init_mps_model():
    """
    Initialize an MPS model, possibly jitted
    """
    # assert JITTED is False

    # Get embedding function
    if EMBED_FUN is not None:
        raw_embed = partial(EMBED_FUN, emb_dim=INPUT_DIM)
        embed_fun = FixedEmbedding(raw_embed, unit_interval)
    else:
        embed_fun = None

    mps = ProbMPS(
        MNIST_SIZE,
        INPUT_DIM,
        BOND_DIM,
        embed_fun=embed_fun,
        complex_params=COMPLEX,
    )
    mps = mps.to(DTYPE)

    # TODO: JIT stuff

    return mps


def mnist_eval(model, data, backward_pass=False):
    """
    Evaluate batches of fake data with same size as MNIST
    """
    for batch in data:
        with record_function("MPS_FORWARD"):
            loss = model.loss(batch, slim_eval=SLIM_EVAL, parallel_eval=PARALLEL_EVAL)

        if backward_pass:
            with record_function("MPS_BACKWARD"):
                loss.backward()
            model.zero_grad()


def profile_mnist_eval(warmup=False):
    """
    Compute the average time needed for batches of fake data
    """
    start = perf_counter()

    # Initialize model and data to evaluate it on
    dataset = init_data()
    model = init_mps_model()
    eval_fun = partial(mnist_eval, backward_pass=INCLUDE_BACKWARD)

    if warmup:
        eval_fun(model, dataset)

    if not FAKE_PROFILE:
        with profile(
            record_shapes=True, profile_memory=PROFILE_MEMORY, with_stack=WITH_STACK
        ) as prof:
            eval_fun(model, dataset)
    else:
        eval_fun(model, dataset)
        prof = None

    total_time = perf_counter() - start
    return prof, total_time


def make_stats_dict(prof, total_time):
    """
    Function condensing full profile object into summarized dictionary
    """

    # Extract sorted event information from the profiler object
    av_obj = prof.key_averages(group_by_stack_n=STACK_DEPTH, group_by_input_shape=True)
    events = sorted(av_obj, key=lambda e: getattr(e, SORT_ATTR), reverse=True)

    # Build the stats dict one event at a time
    stats_dict = {}
    for e in events:
        # All non-callable properties that aren't big or useless
        event_dict = {}
        for a in dir(e):
            attr = getattr(e, a)
            if (
                a.startswith("__")
                or hasattr(attr, "__call__")
                or a in ["cpu_children", "cpu_parent", "key"]
            ):
                continue
            event_dict[a] = attr
        stats_dict[e.key] = event_dict

    stats_dict["wall_clock"] = total_time
    stats_dict["self_cpu_time_total"] = av_obj.self_cpu_time_total
    stats_dict["table"] = av_obj.table(
        sort_by=SORT_ATTR, row_limit=30, max_src_column_width=210
    )

    return stats_dict


# def compare_jitted_vs_not():
#     """
#     Compute the time needed for batches with jitted vs. regular ProbMPS
#     """
#     time_dict, table_dict = {}, {}
#     for jitted in [False, True]:
#         prof_out = profile_mnist_eval(warmup=True)
#         stats_dict = make_stats_dict(*prof_out)

#         time_dict[jitted] = stats_dict["self_cpu_time_total"]
#         table_dict[jitted] = stats_dict["table"]

#     return time_dict, table_dict


@restore_config
def compare_different_configs(
    global_names, config_tuples, compare_at_end=True, printed_rows=15
):
    """
    Profile the ProbMPS model with different experimental configurations

    Args:
        global_names: List of strings containing the global variable names
            which will be varied between profiling runs (e.g. ["BOND_DIM", "COMPLEX"])
        config_tuples: List of tuples giving the values which the global
            variables will be set to (e.g. [(10, False), (15, True)]). The
            length of config_tuples is the number of profiling runs
        compare_at_end: Whether or not to run the compare_tables function on the
            resultant configuration setup before returning values.
        printed_rows: The choice of rows to display if compare_tables is called.

    Returns:
        time_dict: Mapping from config tuples to config runtimes (in seconds)
        time_dict: Mapping from config tuples to profiling tables
    """
    # Promote singleton values to tuples, for convenience
    if all(not isinstance(tup, (tuple, list)) for tup in config_tuples):
        config_tuples = [(v,) for v in config_tuples]

    # Check that global variables are valid, config tuples have right length
    global_names = [n.upper() for n in global_names]
    assert all(n in globals() for n in global_names), global_names
    assert all(len(tup) == len(global_names) for tup in config_tuples)

    time_dict, table_dict = {}, {}
    for tup in config_tuples:
        # Change the global configuration variables, update derived globals
        globals().update({k: v for k, v in zip(global_names, tup)})
        set_derived_globals()

        # Run the experiment and summarize the profiling information
        prof_out = profile_mnist_eval(warmup=True)
        stats_dict = make_stats_dict(*prof_out)

        # Condense this into total runtime and profiling table
        time_dict[tup] = stats_dict["self_cpu_time_total"] / 10e5
        table_dict[tup] = stats_dict["table"]

    if compare_at_end:
        compare_tables(table_dict, printed_rows, global_names)

    return time_dict, table_dict


def compare_tables(table_dict, printed_rows, key_labels):
    """
    Print comparison of some rows profiled for different experimental configurations

    Args:
        table_dict: Dictionary generated by compare_different_configs, whose
            keys are tuples of configuration variables and whose values are
            profiling information for that particular configuration.
        printed_rows: Either a name for some row in the profiling table (e.g.
            "aten::matmul"), a list of such names, or a positive integer. In
            the last case, the first k rows of the table will be displayed,
            for k=printed_rows. Setting printed_rows=-1 will print everything.
        key_labels: List of the names to display for the config variables,
            whose length must agree with the length of the tuples used as keys
            for table_dict.
    """
    # Check that key_labels is in agreement with the keys of table_dict
    if isinstance(key_labels, str):
        key_labels = [key_labels]
    assert all(len(tup) == len(key_labels) for tup in table_dict)

    # Deal with different possible values of printed_rows
    indexing = isinstance(printed_rows, int)
    if indexing:
        num_to_take = printed_rows
        if num_to_take == -1:
            num_to_take = 10e6
    if isinstance(printed_rows, str):
        printed_rows = [printed_rows]
        indexing = False

    # Print table header (first three lines of every table)
    example_table = next(iter(table_dict.values()))
    header = example_table.split("\n")[:3]
    print("\n".join(header))

    # Print out the values for each row, for every config in table_dict
    for tup in table_dict:
        print(", ".join([f"{k}={v}" for k, v in zip(key_labels, tup)]) + ":")
        rows = table_dict[tup].split("\n")[3:]

        for i, r in enumerate(rows):
            if indexing and i < num_to_take:
                print(r)
            if not indexing and any(pr in r for pr in printed_rows):
                print(r)
        print()


if __name__ == "__main__":
    # # Compare different embeddings
    # time_dict, table_dict = compare_different_configs(
    #     ["EMBED_FUN"], [None, trig_embed, legendre_embed]
    # )

    # # Compare different evaluation methods for small bond dims
    # time_dict, table_dict = compare_different_configs(
    #     ["BOND_DIM", "EVAL_TYPE"], [(16, "slim"), (16, "default"), (16, "parallel")]
    # )

    # Compare different evaluation methods for large bond dims
    time_dict, table_dict = compare_different_configs(
        ["BOND_DIM", "EVAL_TYPE"], [(100, "default"), (100, "slim")]
    )

    # # Compare forward+backward passes vs. forward pass alone
    # time_dict, table_dict = compare_different_configs(
    #     ["INCLUDE_BACKWARD"], [True, False]
    # )

    print(time_dict)
