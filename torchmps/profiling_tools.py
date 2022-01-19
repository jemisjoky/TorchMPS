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
NUM_BATCH = 10
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
FAKE_PROFILE = True    # Whether to run code without profiling
PROFILE_MEMORY = True
INCLUDE_BACKWARD = False

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

    if FAKE_PROFILE:
        eval_fun(model, dataset)
        prof = None
    else:
        with profile(record_shapes=True, profile_memory=PROFILE_MEMORY) as prof:
            eval_fun(model, dataset)

    total_time = perf_counter() - start
    return prof, total_time


if __name__ == "__main__":
    jitted = warmup = False

    prof, total_time = profile_mnist_eval(
        jitted=jitted, warmup=warmup, include_backward=INCLUDE_BACKWARD
    )

    if not FAKE_PROFILE:
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        if SAVE_TRACE:
            prof.export_chrome_trace("trace.json")

    print(f"TOTAL_TIME: {total_time:.3f}s")
