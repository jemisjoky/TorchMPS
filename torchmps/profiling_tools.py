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
# MNIST_SIZE = 28 ** 2
# BATCH_SIZE = 128
# NUM_BATCH = 10
INPUT_DIM = 5
BOND_DIM = 20
COMPLEX = False
# EMBED_FUN = trig_embed  # Set to None for discrete modeling
EMBED_FUN = None  # Set to None for discrete modeling
DTYPE = torch.float32
EVAL_TYPE = "slim"  # Options: "slim", "default", "parallel"

# Profiling parameters
SAVE_TRACE = False      # Whether to save trace of program
FAKE_PROFILE = False    # Whether to run code without profiling
PROFILE_MEMORY = True
INCLUDE_BACKWARD = False
SORT_ATTR = "self_cpu_time_total"

# Function for setting derived globals
def fix_globals():
    global SLIM_EVAL, PARALLEL_EVAL
    assert EVAL_TYPE in ["slim", "default", "parallel"]
    SLIM_EVAL = EVAL_TYPE == "slim"
    PARALLEL_EVAL = EVAL_TYPE == "parallel"


fix_globals()


def init_data():
    """
    Generate batches of discrete or continuous fake data of same size as MNIST
    """
    shape = (NUM_BATCH, BATCH_SIZE, MNIST_SIZE)

    if EMBED_FUN is None:
        return torch.randint(INPUT_DIM, shape)
    else:
        return torch.rand(shape)


def init_mps_model(jitted=False):
    """
    Initialize an MPS model, possibly jitted
    """
    assert jitted == False

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


def profile_mnist_eval(jitted=False, warmup=False, include_backward=False):
    """
    Compute the average time needed for batches of fake data
    """
    start = perf_counter()

    # Initialize model and data to evaluate it on
    dataset = init_data()
    model = init_mps_model(jitted=jitted)
    eval_fun = partial(mnist_eval, backward_pass=include_backward)

    if warmup:
        eval_fun(model, dataset)

    if not FAKE_PROFILE:
        with profile(record_shapes=True, profile_memory=PROFILE_MEMORY) as prof:
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
    av_obj = prof.key_averages(group_by_input_shape=True)
    events = sorted(av_obj, key=lambda e: getattr(e, SORT_ATTR), reverse=True)

    # Build the stats dict one event at a time
    stats_dict = {}
    for e in events:
        # All non-callable properties that aren't big or useless
        event_dict = {}
        for a in dir(e):
            attr = getattr(e, a)
            if a.startswith("__") or hasattr(attr, "__call__") or a in ["cpu_children", "cpu_parent", "key"]:
                continue
            event_dict[a] = attr
        stats_dict[e.key] = event_dict

    stats_dict["wall_clock"] = total_time
    stats_dict["self_cpu_time_total"] = av_obj.self_cpu_time_total
    stats_dict["table"] = av_obj.table(sort_by="self_cpu_time_total", row_limit=30)

    return stats_dict


def compare_jitted_vs_not():
    """
    Compute the time needed for batches with jitted vs. regular ProbMPS
    """
    times, tables = {}, {}
    for jitted in [False, True]:
        # prof_out = profile_mnist_eval(jitted=jitted, 
        prof_out = profile_mnist_eval(jitted=False, warmup=True, include_backward=INCLUDE_BACKWARD)
        stats_dict = make_stats_dict(*prof_out)

        times[jitted] = stats_dict["self_cpu_time_total"]
        tables[jitted] = stats_dict["table"]

    return times, tables

def compare_rows(tables, printed_rows=["aten::matmul"], key_label="KEY"):
    """
    Print comparison of certain rows (events) profiled for different experimental configurations
    """
    indexing =  isinstance(printed_rows, int)
    if indexing:
        num_to_take = printed_rows
        assert num_to_take >= 0
    if isinstance(printed_rows, str):
        printed_rows = [printed_rows]
        indexing = False
    
    # Print table header (first three lines of table)
    example_table = next(iter(tables.values()))
    header = example_table.split("\n")[:3]
    print("\n".join(header))

    # Print out the values for each row, for every config in tables
    for key in tables:
        print(f"{key_label} = {key}:")
        rows = tables[key].split("\n")[3:]

        for i, r in enumerate(rows):
            if indexing and i < num_to_take:
                print(r)
            if not indexing and any(pr in r for pr in printed_rows):
                print(r)


if __name__ == "__main__":
    # jitted = warmup = False

    # prof, total_time = profile_mnist_eval(
    #     jitted=jitted, warmup=warmup, include_backward=INCLUDE_BACKWARD
    # )

    # if not FAKE_PROFILE:
    #     print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=30))
    #     if SAVE_TRACE:
    #         prof.export_chrome_trace("trace.json")

    # print(f"TOTAL_TIME: {total_time:.3f}s")

    times, tables = compare_jitted_vs_not()
    compare_rows(tables, 5)
