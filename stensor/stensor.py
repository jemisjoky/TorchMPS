import warnings
import itertools
from functools import wraps, lru_cache, reduce

import torch

from .utils import bad_conversion, tupleize, squeeze_dims, flatten_index, scalar_scale

"""
NOTE: Requires torch version >= 1.7.0
"""

# Function giving the target one-norm of a STensor based on its shape.
# TARGET_SCALE is a sort of module-wise hyperparameter whose choice
# influences the stability of operations on STensor instances
@lru_cache()
def TARGET_SCALE(shape, data_dims):
    assert all(d >= 0 for d in data_dims)
    shape = tuple(shape[i] for i in data_dims)

    # We want to have one_norm(tensor) ~= num_el
    # return torch.log2(torch.prod(torch.tensor(shape)).float())
    return torch.log2(torch.prod(torch.tensor(shape)).float())


### STensor core tools ###


def stensor(
    data, stable_dims=(), dtype=None, device=None, requires_grad=False, pin_memory=False
):
    """
    Constructs a STensor from input data and a partition index placement

    Args:
        data: Initial data for the stensor. Can be a list, tuple,
            NumPy ``ndarray``, scalar, and other types.
        stable_dims: List/tuple of dims of data tensor which are stabilized 
            in output STensor. Choosing stable_dims=() (default) gives
            single scale parameter for data, stable_dims=range(0,...,k-1)
            for k = len(data.shape) gives separate scale parameter for each
            element of data
        dtype (optional): the desired data type of returned tensor.
            Default: if ``None``, infers data type from :attr:`data`.
        device (optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
    """
    # For input STensors, shift stable_dims and/or rescale data
    if isinstance(data, STensor):
        if tuple(stable_dims) == data.stable_dims:
            return data.rescale()
        else:
            return move_sdims(data, stable_dims)

    # Convert data to Pytorch float tensor if it's not already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    data = data.float()
    num_dims = len(data.shape)
    assert all(0 <= d < num_dims for d in stable_dims)

    # Initialize with trivial scale tensor
    shape = data.shape
    s_shape = tuple((d if i in stable_dims else 1) for i, d in enumerate(shape))
    scale = torch.zeros(
        s_shape,
        requires_grad=data.requires_grad,
        dtype=data.dtype,
        layout=data.layout,
        device=data.device,
    )

    # Build and rescale STensor
    stensor = STensor(data, scale)
    stensor.rescale_()
    return stensor


class STensor:
    def __init__(self, data, scale):
        # Check that the shapes of data and scale tensors are compatible
        assert len(data.shape) == len(scale.shape)
        assert all(ss in (1, ds) for ss, ds in zip(scale.shape, data.shape))

        self.data = data
        self.scale = scale

    def __str__(self):
        # Disclaimer for any questionable printed values
        disclaimer = (
            "\ninf and/or zero entries may be artifact of conversion"
            "\nto Tensor, use repr to view underlying data"
        )
        # Check for warning during conversion, remove disclaimer if not needed
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            t = self.torch()
            if len(wrec) == 0:
                disclaimer = ""
        return f"s{t}{disclaimer}"

    def __repr__(self):
        return f"STensor(data=\n{self.data}, scale=\n{self.scale})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def stable_dims(self):
        return tuple(i for i, d in enumerate(self.scale.shape) if d > 1)

    @property
    def data_dims(self):
        return tuple(i for i, d in enumerate(self.scale.shape) if d == 1)

    @property
    def T(self):
        return STensor(self.data.T, self.scale.T)

    def rescale_(self):
        """In-place rescaling method"""
        # Get the L1 norm of data and scale correction for each fiber
        data_dims = self.data_dims
        if data_dims == ():
            tens_scale = self.data.abs()
        else:
            tens_scale = torch.sum(self.data.abs(), dim=data_dims, keepdim=True)
        log_shift = torch.floor(
            TARGET_SCALE(self.shape, data_dims) - torch.log2(tens_scale)
        )

        # Keep the scale for zero fibers unchanged
        if torch.any(torch.isinf(log_shift)):
            log_shift = torch.where(
                torch.isfinite(log_shift), log_shift, torch.zeros_like(log_shift)
            )

        self.data *= 2 ** log_shift
        self.scale -= log_shift

    def rescale(self):
        """Return STensor with rescaled data"""
        # Get the L1 norm of data and scale correction for each fiber
        data_dims = self.data_dims
        if data_dims is ():
            tens_scale = self.data.abs()
        else:
            tens_scale = torch.sum(self.data.abs(), dim=data_dims, keepdim=True)
        log_shift = torch.floor(
            TARGET_SCALE(self.shape, data_dims) - torch.log2(tens_scale)
        )

        # Keep the scale for zero fibers unchanged
        if torch.any(torch.isinf(log_shift)):
            log_shift = torch.where(
                torch.isfinite(log_shift), log_shift, torch.zeros_like(log_shift)
            )

        return STensor(self.data * (2 ** log_shift), self.scale - log_shift)

    def torch(self):
        """Return vanilla Pytorch Tensor reduction of STensor"""
        tensor = self.data * 2 ** self.scale

        # Check for and warn about errors in conversion
        if bad_conversion(self, tensor):
            warnings.warn(
                "Underflow and/or overflow detected " "during torch() call",
                RuntimeWarning,
            )

        return tensor

    def __torch_function__(self, fun, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        type_cond = all(issubclass(t, (torch.Tensor, STensor)) for t in types)
        if fun in STABLE_FUNCTIONS and type_cond:
            return STABLE_FUNCTIONS[fun](*args, **kwargs)
        else:
            print(
                f"STensor version of 'torch.{fun.__name__}' not yet "
                "implemented, let me know at https://github.com/jemisjoky"
                "/STensor/issues if you want me to prioritize this"
            )
            return NotImplemented

    # The rest of the magic methods

    def __pow__(self, other):
        return STensor(self.data ** other, self.scale * other).rescale()

    def __matmul__(self, other):
        return torch.matmul(self, other)

    def __rmatmul__(self, other):
        return torch.matmul(other, self)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        # Multiplication is commutative, so __rmul__ == __mul__
        return self.mul(other)

    def __imul__(self, other):
        self.mul_(other)

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        # Addition is commutative, so __radd__ == __add__
        return self.add(other)

    def __iadd__(self, other):
        self.add_(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return torch.sub(other, self)

    def __isub__(self, other):
        self.sub_(other)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        return torch.div(other, self)

    def __itruediv__(self, other):
        self.div_(other)

    def __neg__(self):
        return STensor(-self.data, self.scale)

    def __floordiv__(self, other):
        return torch.floor(self.div(other))

    def __rfloordiv__(self, other):
        return torch.floor(torch.div(other, self))

    def __ifloordiv__(self, other):
        raise RuntimeError(
            "In-place floor division doesn't really make "
            "sense with STensors, integer precision isn't guaranteed"
        )

    # def __mod__(self, other):
    #     pass

    def __abs__(self):
        return self.abs()

    def __eq__(self, other):
        return self.eq(other)

    def __gt__(self, other):
        return self.gt(other)

    def __lt__(self, other):
        return self.lt(other)

    def __ge__(self, other):
        return self.ge(other)

    def __le__(self, other):
        return self.le(other)

    def __len__(self):
        return len(self.data)

    def __contains__(self, other):
        return self.torch().__contains__(other)

    def __reversed__(self):
        return STensor(self.data.flip(0), self.scale.flip(0))

    def __bool__(self):
        return self.data.__bool__()

    def __getitem__(self, idx):
        if isinstance(idx, (torch.Tensor, STensor)):
            raise NotImplementedError("STensors don't support advanced indexing yet")

        # Produce "flat" index to use for indexing scale tensor
        flat_idx = flatten_index(idx, self.scale)
        stens_out = STensor(self.data[idx], self.scale[flat_idx])
        stens_out.rescale_()

        return stens_out

    def __setitem__(self, idx, value):
        # TODO: Fix this limitation to scalar scale tensors
        if not scalar_scale(self) or (
            isinstance(value, STensor) and not scalar_scale(value)
        ):
            raise NotImplementedError(
                "Assignment currently only implemented "
                "for stensors with single-element scale tensors"
            )

        # Ensure value is stensor or tensor
        if not isinstance(value, (torch.Tensor, STensor)):
            value = torch.tensor(value)

        # Adapt value to have same scale tensor as self
        if isinstance(value, STensor):
            shift = value.scale.view(()) - self.scale.view(())
            value = value.data
        else:
            shift = -self.scale.view(())
        value = value * 2 ** shift

        # Set data tensor
        self.data[idx] = value
        self.rescale_()


def move_sdims(stens, stable_dims):
    """Return copy of input STensor with new stable dims"""
    # Get the data dimensions associated with new stable dims
    assert all(0 <= i < stens.ndim for i in stable_dims)
    data_dims = tuple(i for i in range(stens.ndim) if i not in stable_dims)

    # Rescale data tensor relative to maximum of scale values, expanding
    # the slices of the former and getting a preliminary scale
    new_scale = torch.amax(stens.scale, dim=data_dims, keepdim=True)
    new_data = stens.data * 2 ** (stens.scale - new_scale)

    # Get the norm of all new slices as a correction to the above scale
    if data_dims == ():
        new_norms = new_data.abs()
    else:
        new_norms = torch.sum(new_data.abs(), dim=data_dims, keepdim=True)
    correction = torch.floor(
        TARGET_SCALE(new_data.shape, data_dims) - torch.log2(new_norms)
    )

    # Filter out any spurious infinities from zero slices
    if torch.any(torch.isinf(correction)):
        correction = torch.where(
            torch.isfinite(correction), correction, torch.zeros_like(correction)
        )

    # Apply correction to new scale and data tensors, return result
    new_data *= 2 ** correction
    new_scale = new_scale - correction
    new_stens = STensor(new_data, new_scale)
    assert new_stens.stable_dims == stable_dims
    new_stens.rescale_()
    return new_stens


def same_scale(*tensors):
    """
    Convert tensors into list of data tensors with a common scale tensor

    Args:
        tensors:   Any number of STensors, Pytorch Tensors, or anything 
                   that can be converted into such datatypes. All inputs 
                   must be jointly broadcastable

    Output:
        data_list: List of Pytorch Tensors with the same shape
        out_scale: Tensor giving the scale of all entries of data_list
    """
    # For each input, pull out data tensor and, if possible, scale tensor
    data_list, scale_list = [], []
    for t in tensors:
        if isinstance(t, STensor):
            data_list.append(t.data)
            scale_list.append(t.scale)
        else:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            data_list.append(t)
            scale_list.append(torch.zeros((1,) * len(t.shape)))

    # Broadcast data and scale tensors
    data_list = torch.broadcast_tensors(*data_list)
    scale_list = torch.broadcast_tensors(*scale_list)

    # Get shared scale tensor, which is elementwise max of input scales
    if len(tensors) == 2:
        out_scale = torch.maximum(*scale_list)
    else:
        out_scale = reduce(torch.maximum, scale_list[1:], scale_list[0])

    # Rescale all data tensors to correspond to out_scale
    data_list = [t * 2 ** (s - out_scale) for t, s in zip(data_list, scale_list)]

    return tuple(data_list), out_scale


### Re-implementations of individual Pytorch functions ###

# Dictionary to store reimplemented Pytorch functions for use on stensors
STABLE_FUNCTIONS = {}


def minimal_wrap(new_fun):
    """Take a single function and set it as the Torch override for STensors"""
    fun_name = new_fun.__name__
    assert fun_name in dir(torch)
    torch_fun = getattr(torch, fun_name)
    new_fun.__doc__ = torch_fun.__doc__
    STABLE_FUNCTIONS[torch_fun] = new_fun
    return new_fun


@minimal_wrap
def transpose(input, dim0, dim1):
    return STensor(
        torch.transpose(input.data, dim0, dim1),
        torch.transpose(input.scale, dim0, dim1),
    )


@wraps(torch.Tensor.transpose_)
def transpose_(self, dim0, dim1):
    self.data.transpose_(dim0, dim1)
    self.scale.transpose_(dim0, dim1)


STensor.transpose_ = transpose_


@wraps(torch.Tensor.view)
def view(self, *shape):
    # TODO: Handle case where self has nontrivial stable dims
    if not scalar_scale(self):
        raise NotImplementedError(
            "STensor.view currently only implemented "
            "for stensors with single-element scale tensors"
        )
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    return STensor(self.data.view(*shape), self.scale.view((-1,) * len(shape)))


STensor.view = view


@wraps(torch.Tensor.view_as)
def view_as(self, other):
    # TODO: When other is an STensor, shift stable dims to match other
    return self.view(other.shape)


STensor.view_as = view_as


@minimal_wrap
def reshape(input, *shape):
    # TODO: Handle case where input has nontrivial stable dims
    if not scalar_scale(input):
        raise NotImplementedError(
            "reshape currently only implemented "
            "for stensors with single-element scale tensors"
        )
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    return STensor(input.data.reshape(*shape), input.scale.view((-1,) * len(shape)))


@minimal_wrap
def mv(input, vec):
    # Ensure inputs are stensors with correct data_dims
    if not isinstance(input, STensor):
        input = stensor(input)
    if not isinstance(vec, STensor):
        vec = stensor(vec)
    if input.stable_dims != () or vec.stable_dims != ():
        raise ValueError(f"Both inputs to mv must have trivial stable dims")

    # Apply mv on underlying data, sum scale values
    output = STensor(torch.mv(input.data, vec.data), input.scale[0] + vec.scale)
    output.rescale_()

    return output


### Tools to convert families of Pytorch functions to ones on STensors ###


def hom_wrap(fun_name, hom_degs, data_lens, in_place=False):
    """
    Wrapper for reasonably simple homogeneous Pytorch functions

    Args:
        fun_name:  Name of homogeneous Pytorch function to be wrapped
        hom_degs:  List of integers, each giving the degree of homoegeneity 
                   of a homogeneous input argument to fun_name
        data_lens: List of non-negative integers, each giving the minimum 
                   number of data dims at the end of homog input argument
        in_place:  Boolean specifying if we should implement the operation 
                   as an in-place method
    """
    # TODO: Simplify code to reflect data_lens being uniform for all args
    # Preprocess homogeneous info to get flags to be called by stable_fun
    assert all(d >= 0 for d in data_lens)
    num_homs = len(hom_degs)
    if in_place:
        torch_fun = getattr(torch.Tensor, fun_name)
    else:
        torch_fun = getattr(torch, fun_name)

    @wraps(torch_fun)
    def stable_fun(*args, **kwargs):
        # Separate out homogeneous args and put everything in all_args
        all_args, in_scales = [], []
        for i, t in enumerate(args):
            if i < num_homs:
                # Homogeneous input args
                if isinstance(t, STensor):
                    all_args.append(t.data)
                    in_scales.append(hom_degs[i] * t.scale)

                    # Check that homog op is acting only on data dims, and
                    # not on stable dims
                    nd, dd = t.ndim, t.data_dims
                    if not all(j in dd for j in range(nd - data_lens[i], nd)):
                        raise ValueError(
                            f"STensor input {i} to {fun_name} "
                            f"must have last {data_lens[i]} dims as data "
                            f"dims, current data dims are {t.data_dims}"
                        )
                else:
                    # Nonhomogeneous input args
                    all_args.append(t)
            else:
                # Other input args, which can be arbitrary
                all_args.append(t)

        # Compute overall rescaling associated with input tensors
        if len(in_scales) > 1:
            out_scale = sum(torch.broadcast_tensors(*in_scales))
        else:
            out_scale = in_scales.pop()

        # Call wrapped Pytorch function, get output as list, and return
        # Different behavior for in-place vs regular cases
        if in_place:
            # Call in-place method of data tensor, then readjust scale
            self = args[0]  # <- Object whose method is being called
            getattr(self.data, fun_name)(*all_args[1:], **kwargs)
            self.scale = out_scale
            self.rescale_()
        else:
            # Call Torch function with data, then convert to stensor
            output = torch_fun(*all_args, **kwargs)
            assert isinstance(output, torch.Tensor)
            stens = STensor(output, out_scale)
            stens.rescale_()
            return stens

    return stable_fun


def torch_wrap(fun_name, in_place=False, data_only=False, rescale=False):
    """
    Wrapper for black-box Pytorch functions

    Args:
        fun_name:   Name of Pytorch function to be wrapped
        in_place:   Boolean specifying if we should implement the 
                    operation as an in-place method
        data_only:  Whether to call function on data attribute alone, 
                    rather than rescaled data, and to later return 
                    unwrapped tensor
        rescale:    Whether the first two arguments should be rescaled to
                    have the same scale tensor, which is then used as
                    a scale tensor for the output when data_only is True
    """
    # TODO: Refactor this to minimize all the spaghetti code
    assert not (in_place and data_only)
    if in_place:
        torch_fun = getattr(torch.Tensor, fun_name)
    else:
        torch_fun = getattr(torch, fun_name)

    @wraps(torch_fun)
    def wrapped_fun(*args, **kwargs):
        # For in-place evaluation, pull out first input
        if in_place:
            self = args[0]  # <- Object whose method is being called
            assert isinstance(self, STensor), (
                "Cannot call in-place torch.Tensor " "methods with STensors as input"
            )

        # Rescale first two input arguments if need be
        # NOTE: If a new method is added to SCALE_TORCH, revise this
        args = list(args)
        if rescale:
            data_list, out_scale = same_scale(*args[:2])
            args[:2] = [STensor(t, out_scale) for t in data_list]

        # Replace any STensor args with Torch tensors
        data_list, scale_list = [], []
        for i, t in enumerate(args):
            if isinstance(t, STensor):
                data_list.append(
                    t.data if (data_only or (rescale and i < 2)) else t.torch()
                )
                scale_list.append(t.scale)
            else:
                data_list.append(t)

        assert len(scale_list) > 0, (
            "STensor input must appear within " f"positional arguments of {fun_name}"
        )

        # Compute overall rescaling associated with input tensors
        if not data_only:
            try:
                scale_shape = torch.broadcast_tensors(*scale_list)[0].shape
            except:
                raise GENERIC_ERROR

        # Call wrapped Pytorch function, get output as list, and return.
        # Different behavior required for in-place vs regular cases
        if in_place:
            # Update the data and scale attributes of self
            self.data = data_list[0]
            if rescale:
                self.scale = scale_list[0]
            else:
                self.scale.zero_()

            # Call in-place method of data tensor, then readjust scale
            getattr(self.data, fun_name)(*data_list[1:], **kwargs)
            self.rescale_()
        else:
            # Call Torch function with data, then convert to stensor
            output = torch_fun(*data_list, **kwargs)
            assert isinstance(output, torch.Tensor) or data_only

            if data_only:
                return output
            else:
                if len(scale_shape) != len(output.shape):
                    raise GENERIC_ERROR

                if rescale:
                    try:
                        stens = STensor(output, out_scale)
                    except:
                        raise GENERIC_ERROR
                else:
                    stable_dims = tuple(i for i, d in enumerate(scale_shape) if d > 1)
                    stens = stensor(output, stable_dims)
                stens.rescale_()
                return stens

    return wrapped_fun


def log_wrap(fun_name):
    """Simple wrapper to reimplement and register logarithm functions"""
    assert fun_name in ["log", "log10", "log2"]
    torch_fun = getattr(torch, fun_name)
    base_lookup = {
        "log": torch.log2(torch.exp(torch.ones(()))),
        "log10": torch.log2(torch.tensor(10.0)),
        "log2": torch.ones(()),
    }
    base_coeff = base_lookup[fun_name]

    # The new logarithm function
    @wraps(torch_fun)
    def log_fun(input):
        assert isinstance(input, STensor)
        output = torch_fun(input.data) + input.scale / base_coeff
        return stensor(output)

    # Register the new logarithm function
    STABLE_FUNCTIONS[torch_fun] = log_fun


def sumlike_wrap(fun_name):
    """Handle torch.sum and torch.mean"""
    # Define appropriate torch function, the rest of the logic is the same
    assert fun_name in ["sum", "mean"]
    torch_fun = getattr(torch, fun_name)

    @wraps(torch_fun)
    def sumlike_fun(input, dim=None, keepdim=False):
        nodim = dim is None
        if nodim:
            # Remove stable dims, then sum over data
            input = move_sdims(input, ())
            data_sum = torch_fun(input.data)
            scale = input.scale.view(())
            output = STensor(data_sum, scale)
        else:
            # Convert dim to list of non-negative indices to sum over
            dim_list = tupleize(dim, input.ndim)
            # Make summed indices data dims, then sum over data tensor
            new_sdims = tuple(i for i in input.stable_dims if i not in dim_list)
            input = move_sdims(input, new_sdims)
            data_sum = torch_fun(input.data, dim, keepdim=keepdim)
            scale = input.scale
            if not keepdim:
                scale = squeeze_dims(scale, dim_list)
            output = STensor(data_sum, scale)
        output.rescale_()
        return output

    # Register the new sum-like function
    STABLE_FUNCTIONS[torch_fun] = sumlike_fun


def existing_method_from_name(fun_name):
    """Add method to STensor and torch.Tensor for existing stable function"""
    global STensor
    assert hasattr(torch.Tensor, fun_name)
    if getattr(torch, fun_name) in STABLE_FUNCTIONS:
        stable_fun = STABLE_FUNCTIONS[getattr(torch, fun_name)]
        STABLE_FUNCTIONS[getattr(torch.Tensor, fun_name)] = stable_fun
        setattr(STensor, fun_name, stable_fun)
    else:
        print(f"STILL NEED TO IMPLEMENT {fun_name}")


def inplace_hom_method_from_name(fun_name):
    """Add in-place versions of homogeneous function to STensor"""
    assert hasattr(torch.Tensor, fun_name)
    assert fun_name[-1] == "_"
    hom_info = HOMOG[fun_name[:-1]]
    hom_degs, data_len = zip(*hom_info)
    stable_method = hom_wrap(fun_name, hom_degs, data_len, in_place=True)
    setattr(STensor, fun_name, stable_method)


def inplace_torch_method_from_name(fun_name):
    """Add in-place versions of simple torch function to STensor"""
    assert hasattr(torch.Tensor, fun_name)
    assert fun_name[-1] == "_"
    try:
        data_only = TORCH[fun_name[:-1]]
    except:
        data_only = SCALE_TORCH[fun_name[:-1]]
    wrapped_method = torch_wrap(fun_name, in_place=True, data_only=data_only)
    setattr(STensor, fun_name, wrapped_method)


### Re-registration of the Pytorch library as stable functions ###

# Each value gives tuple of (hom_deg, data_len) pairs (one for each arg),
# where hom_deg gives the degree of homogeneity and data_len gives the
# minimum number of data indices for that argument
HOMOG = {
    "abs": ((1, 0),),
    "bmm": ((1, 2), (1, 2)),
    "conj": ((1, 0),),
    "cosine_similarity": ((0, 1), (0, 1)),
    "div": ((1, 0), (-1, 0)),
    "dot": ((1, 1), (1, 1)),
    "ger": ((1, 1), (1, 1)),
    "imag": ((1, 0),),
    "real": ((1, 0),),
    "inverse": ((-1, 2),),
    "matmul": ((1, 1), (1, 1)),
    "mm": ((1, 2), (1, 2)),
    "mul": ((1, 0), (1, 0)),
    "pinverse": ((-1, 2),),
    "reciprocal": ((-1, 0),),
    "relu": ((1, 0),),
    "square": ((2, 2),),
    "t": ((1, 2),),
    "trace": ((1, 2),),
    "true_divide": ((-1, 0),),
}

# Values give data_only flag sent to torch_wrap
TORCH = {
    "acos": False,
    "angle": False,
    "asin": False,
    "atan": False,
    "atan2": False,
    "cartesian_prod": False,
    "ceil": False,
    "celu": False,
    "clamp": False,
    "clamp_max": False,
    "clamp_min": False,
    "conv1d": False,
    "conv2d": False,
    "conv3d": False,
    "conv_tbc": False,
    "conv_transpose1d": False,
    "conv_transpose2d": False,
    "conv_transpose3d": False,
    "cos": False,
    "cosh": False,
    "digamma": False,
    "erf": False,
    "erfc": False,
    "erfinv": False,
    "exp": False,
    "expm1": False,
    "fft": False,
    "floor": False,
    "frac": False,
    "fmod": False,
    "ifft": False,
    "irfft": False,
    "isfinite": True,
    "isinf": True,
    "isnan": True,
    "is_complex": True,
    "is_floating_point": True,
    "is_nonzero": True,
    "is_same_size": True,
    "kthvalue": False,
    "lerp": False,
    "lgamma": False,
    "log1p": False,
    "logdet": False,
    "logical_and": True,
    "logical_not": True,
    "logical_or": True,
    "logical_xor": True,
    "logsumexp": False,
    "numel": True,
    "rfft": False,
    "round": False,
    "sigmoid": False,
    "sign": True,
    "sin": False,
    "sinh": False,
    "stft": False,
    "tan": False,
    "tanh": False,
    "trunc": False,
}

# Values give data_only flag sent to torch_wrap
# Change torch_wrap if I add a function that doesn't take exactly 2 inputs
SCALE_TORCH = {
    "add": False,
    "sub": False,
    "eq": True,
    "ne": True,
    "equal": True,
    "ge": True,
    "gt": True,
    "le": True,
    "lt": True,
}

LOG_FUNS = ["log", "log10", "log2"]
SUMLIKE_FUNS = ["sum", "mean"]
MODELIKE_FUNS = ["mode", "median"]
VARLIKE_FUNS = ["var", "std"]

# Register all homogeneous functions as functions on stensors
for fun_name, hom_data in HOMOG.items():
    hom_degs, data_lens = zip(*hom_data)
    torch_fun = getattr(torch, fun_name)
    assert torch_fun not in STABLE_FUNCTIONS
    stable_fun = hom_wrap(fun_name, hom_degs, data_lens, in_place=False)
    STABLE_FUNCTIONS[torch_fun] = stable_fun

# Register simple Pytorch functions as functions on stensors
for fun_name, data_only in TORCH.items():
    torch_fun = getattr(torch, fun_name)
    assert torch_fun not in STABLE_FUNCTIONS
    wrapped_fun = torch_wrap(
        fun_name, in_place=False, data_only=data_only, rescale=False
    )
    STABLE_FUNCTIONS[torch_fun] = wrapped_fun

# Register scaled Pytorch functions as functions on stensors
for fun_name, data_only in SCALE_TORCH.items():
    torch_fun = getattr(torch, fun_name)
    assert torch_fun not in STABLE_FUNCTIONS
    wrapped_fun = torch_wrap(
        fun_name, in_place=False, data_only=data_only, rescale=True
    )
    STABLE_FUNCTIONS[torch_fun] = wrapped_fun

# Register logarithm functions as functions on stensors
for fun_name in LOG_FUNS:
    log_wrap(fun_name)

# Register sum and mean
for fun_name in SUMLIKE_FUNS:
    sumlike_wrap(fun_name)

# Doesn't make sense with stensors, and/or method doesn't have PyTorch
# documentation. Not implementing
DOUBTFUL = [
    "affine_grid_generator",
    "alpha_dropout",
    "adaptive_avg_pool1d",
    "adaptive_max_pool1d",
    "avg_pool1d",
    "batch_norm",
    "batch_norm_backward_elemt",
    "batch_norm_backward_reduce",
    "batch_norm_elemt",
    "batch_norm_gather_stats",
    "batch_norm_gather_stats_with_counts",
    "batch_norm_stats",
    "batch_norm_update_stats",
    "bernoulli",
    "bilinear",
    "binary_cross_entropy_with_logits",
    "bincount",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "constant_pad_nd",
    "convolution",
    "cosine_embedding_loss",
    "ctc_loss",
    "dequantize",
    "dropout",
    "dsmm",
    "embedding",
    "embedding_bag",
    "empty_like",
    "fake_quantize_per_channel_affine",
    "fake_quantize_per_tensor_affine",
    "fbgemm_linear_fp16_weight",
    "fbgemm_linear_fp16_weight_fp32_activation",
    "fbgemm_linear_int8_weight",
    "fbgemm_linear_int8_weight_fp32_activation",
    "fbgemm_linear_quantize_weight",
    "fbgemm_pack_gemm_matrix_fp16",
    "fbgemm_pack_quantized_matrix",
    "feature_alpha_dropout",
    "feature_dropout",
    "frobenius_norm",
    "geqrf",
    "grid_sampler",
    "grid_sampler_2d",
    "grid_sampler_3d",
    "group_norm",
    "gru",
    "gru_cell",
    "hardshrink",
    "hinge_embedding_loss",
    "histc",
    "hsmm",
    "hspmm",
    "index_add",
    "index_copy",
    "index_fill",
    "index_put",
    "index_select",
    "instance_norm",
    "int_repr",
    "is_distributed",
    "is_signed",
    "isclose",
    "kl_div",
    "layer_norm",
    "lobpcg",
    "log_softmax",
    "lstm",
    "lstm_cell",
    "margin_ranking_loss",
    "masked_fill",
    "masked_scatter",
    "masked_select",
    "matrix_rank",
    "max_pool1d",
    "max_pool1d_with_indices",
    "max_pool2d",
    "max_pool3d",
    "meshgrid",
    "miopen_batch_norm",
    "miopen_convolution",
    "miopen_convolution_transpose",
    "miopen_depthwise_convolution",
    "miopen_rnn",
    "multinomial",
    "mvlgamma",
    "native_batch_norm",
    "native_layer_norm",
    "native_norm",
    "neg",
    "nonzero",
    "norm",
    "norm_except_dim",
    "normal",
    "nuclear_norm",
    "ones_like",
    "pairwise_distance",
    "pixel_shuffle",
    "poisson",
    "poisson_nll_loss",
    "polygamma",
    "prelu",
    "q_per_channel_axis",
    "q_per_channel_scales",
    "q_per_channel_zero_points",
    "q_scale",
    "q_zero_point",
    "quantize_per_channel",
    "quantize_per_tensor",
    "quantized_batch_norm",
    "quantized_gru",
    "quantized_gru_cell",
    "quantized_lstm",
    "quantized_lstm_cell",
    "quantized_max_pool2d",
    "quantized_rnn_relu_cell",
    "quantized_rnn_tanh_cell",
    "rand_like",
    "randint_like",
    "randn_like",
    "remainder",
    "renorm",
    "repeat_interleave",
    "result_type",
    "rnn_relu",
    "rnn_relu_cell",
    "rnn_tanh",
    "rnn_tanh_cell",
    "rot90",
    "rrelu",
    "rsub",
    "saddmm",
    "scalar_tensor",
    "scatter",
    "scatter_add",
    "select",
    "selu",
    "smm",
    "softmax",
    "split_with_sizes",
    "spmm",
    "sspaddmm",
    "threshold",
    "topk",
    "tril_indices",
    "triu_indices",
    "triplet_margin_loss",
    "unbind",
    "zeros_like",
]

# Important and/or easy functions
DO_NOW = ["einsum", "max", "min", "sqrt", "rsqrt", "median", "mode", "var", "std"]

# Somewhat important and/or trickier functions
DO_SOON = [
    "allclose",
    "argsort",
    "broadcast_tensors",
    "cat",
    "stack",
    "chain_matmul",
    "cumprod",
    "detach",
    "diag",
    "diagonal",
    "flatten",
    "flip",
    "floor_divide",
    "gather",
    "pow",
    "squeeze",
    "unsqueeze",
    "sort",
    "split",
    "dist",
    # Homogeneous functions that for one reason or another can't be handled by hom_wrap
    "pdist",
    "trapz",
    "take",
    "unique_consecutive",
    "var_mean",
    "lu_solve",
    "std_mean",
    "tril",
    "triu",
    "argmax",
    "argmin",
    # Homogeneous functions whose hom degree is based on other info
    "det",
    "prod",
    "matrix_power",
    # Almost a standard homogeneous function, but contains additional dim information
    "tensordot",
    "cross",
    "cumsum",
    # Matrix decompositions whose return types must be respected
    "qr",
    "eig",
    "lstsq",
    "svd",
    "symeig",
    "triangular_solve",
    "solve",
]

# Not important, could be tough
LATER = [
    "addbmm",
    "addcdiv",
    "addcmul",
    "addmm",
    "addmv",
    "addr",
    "baddbmm",
    "cdist",
    "cholesky",
    "cholesky_inverse",
    "cholesky_solve",
    "chunk",
    "clone",
    "combinations",
    "cummax",
    "cummin",
    "diag_embed",
    "diagflat",
    "full_like",
    "narrow",
    "orgqr",
    "ormqr",
    "roll",
    "slogdet",
    "where",
]

ALL_FUN = list(HOMOG.keys()) + DOUBTFUL + list(TORCH) + DO_NOW + DO_SOON + LATER

### Register stabilized Pytorch functions as methods for stensors ###

EXISTING_METHOD = [
    "abs",
    "bmm",
    "conj",
    "div",
    "dot",
    "ger",
    "inverse",
    "matmul",
    "mm",
    "mul",
    "mv",
    "pinverse",
    "reciprocal",
    "relu",
    "square",
    "sum",
    "t",
    "trace",
    "transpose",
    "true_divide",
    "acos",
    "angle",
    "asin",
    "atan",
    "atan2",
    "ceil",
    "clamp",
    "clamp_max",
    "clamp_min",
    "cos",
    "cosh",
    "digamma",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "expm1",
    "fft",
    "floor",
    "fmod",
    "frac",
    "ifft",
    "irfft",
    "is_complex",
    "is_floating_point",
    "is_nonzero",
    "is_same_size",
    "kthvalue",
    "lerp",
    "lgamma",
    "logdet",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "numel",
    "rfft",
    "round",
    "sigmoid",
    "sin",
    "sinh",
    "stft",
    "tan",
    "tanh",
    "trunc",
    "add",
    "sub",
    "eq",
    "ne",
    "equal",
    "ge",
    "gt",
    "le",
    "lt",
    "log",
    "log10",
    "log2",
    "transpose",
    "mean",
    "sum",
    "reshape",
]

HOM_INPLACE = [
    "abs_",
    "div_",
    "mul_",
    "reciprocal_",
    "relu_",
    "square_",
    "t_",
    "true_divide_",
]

TORCH_INPLACE = [
    "acos_",
    "asin_",
    "atan2_",
    "atan_",
    "clamp_",
    "clamp_max_",
    "clamp_min_",
    "cos_",
    "cosh_",
    "digamma_",
    "erf_",
    "erfc_",
    "erfinv_",
    "exp_",
    "expm1_",
    "fmod_",
    "frac_",
    "lerp_",
    "lgamma_",
    "sigmoid_",
    "sin_",
    "sinh_",
    "tan_",
    "tanh_",
    "trunc_",
    "ceil_",
    "floor_",
    "round_",
    "add_",
    "sub_",
]

# Add all methods which I've already implement as stable functions
for name in EXISTING_METHOD:
    existing_method_from_name(name)

# Implement in-place versions of homogeneous functions
for name in HOM_INPLACE:
    inplace_hom_method_from_name(name)

# Implement in-place versions of simple torch functions
for name in TORCH_INPLACE:
    inplace_torch_method_from_name(name)

ATTRIBUTES = [
    "T",
    "__abs__",
    "__add__",
    "__and__",
    "__array__",
    "__array_priority__",
    "__array_wrap__",
    "__bool__",
    "__class__",
    "__complex__",
    "__contains__",
    "__deepcopy__",
    "__delattr__",
    "__delitem__",
    "__dict__",
    "__dir__",
    "__div__",
    "__doc__",
    "__eq__",
    "__float__",
    "__floordiv__",
    "__format__",
    "__ge__",
    "__getattribute__",
    "__getitem__",
    "__gt__",
    "__hash__",
    "__iadd__",
    "__iand__",
    "__idiv__",
    "__ifloordiv__",
    "__ilshift__",
    "__imul__",
    "__index__",
    "__init__",
    "__init_subclass__",
    "__int__",
    "__invert__",
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__iter__",
    "__itruediv__",
    "__ixor__",
    "__le__",
    "__len__",
    "__long__",
    "__lshift__",
    "__lt__",
    "__matmul__",
    "__mod__",
    "__module__",
    "__mul__",
    "__ne__",
    "__neg__",
    "__new__",
    "__nonzero__",
    "__or__",
    "__pow__",
    "__radd__",
    "__rdiv__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__reversed__",
    "__rfloordiv__",
    "__rmul__",
    "__rpow__",
    "__rshift__",
    "__rsub__",
    "__rtruediv__",
    "__setattr__",
    "__setitem__",
    "__setstate__",
    "__sizeof__",
    "__str__",
    "__sub__",
    "__subclasshook__",
    "__torch_function__",
    "__truediv__",
    "__weakref__",
    "__xor__",
    "_backward_hooks",
    "_base",
    "_cdata",
    "_coalesced_",
    "_dimI",
    "_dimV",
    "_grad",
    "_grad_fn",
    "_indices",
    "_is_view",
    "_make_subclass",
    "_nnz",
    "_update_names",
    "_values",
    "_version",
    "abs",
    "abs_",
    "absolute",
    "absolute_",
    "acos",
    "acos_",
    "acosh",
    "acosh_",
    "add",
    "add_",
    "addbmm",
    "addbmm_",
    "addcdiv",
    "addcdiv_",
    "addcmul",
    "addcmul_",
    "addmm",
    "addmm_",
    "addmv",
    "addmv_",
    "addr",
    "addr_",
    "align_as",
    "align_to",
    "all",
    "allclose",
    "amax",
    "amin",
    "angle",
    "any",
    "apply_",
    "arccos",
    "arccos_",
    "arccosh",
    "arccosh_",
    "arcsin",
    "arcsin_",
    "arcsinh",
    "arcsinh_",
    "arctan",
    "arctan_",
    "arctanh",
    "arctanh_",
    "argmax",
    "argmin",
    "argsort",
    "as_strided",
    "as_strided_",
    "as_subclass",
    "asin",
    "asin_",
    "asinh",
    "asinh_",
    "atan",
    "atan2",
    "atan2_",
    "atan_",
    "atanh",
    "atanh_",
    "backward",
    "baddbmm",
    "baddbmm_",
    "bernoulli",
    "bernoulli_",
    "bfloat16",
    "bincount",
    "bitwise_and",
    "bitwise_and_",
    "bitwise_not",
    "bitwise_not_",
    "bitwise_or",
    "bitwise_or_",
    "bitwise_xor",
    "bitwise_xor_",
    "bmm",
    "bool",
    "byte",
    "cauchy_",
    "ceil",
    "ceil_",
    "char",
    "cholesky",
    "cholesky_inverse",
    "cholesky_solve",
    "chunk",
    "clamp",
    "clamp_",
    "clamp_max",
    "clamp_max_",
    "clamp_min",
    "clamp_min_",
    "clip",
    "clip_",
    "clone",
    "coalesce",
    "conj",
    "contiguous",
    "copy_",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "count_nonzero",
    "cpu",
    "cross",
    "cuda",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "data",
    "data_ptr",
    "deg2rad",
    "deg2rad_",
    "dense_dim",
    "dequantize",
    "det",
    "detach",
    "detach_",
    "device",
    "diag",
    "diag_embed",
    "diagflat",
    "diagonal",
    "digamma",
    "digamma_",
    "dim",
    "dist",
    "div",
    "div_",
    "divide",
    "divide_",
    "dot",
    "double",
    "dtype",
    "eig",
    "element_size",
    "eq",
    "eq_",
    "equal",
    "erf",
    "erf_",
    "erfc",
    "erfc_",
    "erfinv",
    "erfinv_",
    "exp",
    "exp2",
    "exp2_",
    "exp_",
    "expand",
    "expand_as",
    "expm1",
    "expm1_",
    "exponential_",
    "fft",
    "fill_",
    "fill_diagonal_",
    "fix",
    "fix_",
    "flatten",
    "flip",
    "fliplr",
    "flipud",
    "float",
    "floor",
    "floor_",
    "floor_divide",
    "floor_divide_",
    "fmod",
    "fmod_",
    "frac",
    "frac_",
    "gather",
    "gcd",
    "gcd_",
    "ge",
    "ge_",
    "geometric_",
    "geqrf",
    "ger",
    "get_device",
    "grad",
    "grad_fn",
    "greater",
    "greater_",
    "greater_equal",
    "greater_equal_",
    "gt",
    "gt_",
    "half",
    "hardshrink",
    "has_names",
    "heaviside",
    "heaviside_",
    "histc",
    "hypot",
    "hypot_",
    "i0",
    "i0_",
    "ifft",
    "imag",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_fill",
    "index_fill_",
    "index_put",
    "index_put_",
    "index_select",
    "indices",
    "int",
    "int_repr",
    "inverse",
    "irfft",
    "is_coalesced",
    "is_complex",
    "is_contiguous",
    "is_cuda",
    "is_distributed",
    "is_floating_point",
    "is_leaf",
    "is_meta",
    "is_mkldnn",
    "is_nonzero",
    "is_pinned",
    "is_quantized",
    "is_same_size",
    "is_set_to",
    "is_shared",
    "is_signed",
    "is_sparse",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "istft",
    "item",
    "kthvalue",
    "layout",
    "lcm",
    "lcm_",
    "le",
    "le_",
    "lerp",
    "lerp_",
    "less",
    "less_",
    "less_equal",
    "less_equal_",
    "lgamma",
    "lgamma_",
    "log",
    "log10",
    "log10_",
    "log1p",
    "log1p_",
    "log2",
    "log2_",
    "log_",
    "log_normal_",
    "log_softmax",
    "logaddexp",
    "logaddexp2",
    "logcumsumexp",
    "logdet",
    "logical_and",
    "logical_and_",
    "logical_not",
    "logical_not_",
    "logical_or",
    "logical_or_",
    "logical_xor",
    "logical_xor_",
    "logit",
    "logit_",
    "logsumexp",
    "long",
    "lstsq",
    "lt",
    "lt_",
    "lu",
    "lu_solve",
    "map2_",
    "map_",
    "masked_fill",
    "masked_fill_",
    "masked_scatter",
    "masked_scatter_",
    "masked_select",
    "matmul",
    "matrix_exp",
    "matrix_power",
    "max",
    "maximum",
    "mean",
    "median",
    "min",
    "minimum",
    "mm",
    "mode",
    "movedim",
    "mul",
    "mul_",
    "multinomial",
    "multiply",
    "multiply_",
    "mv",
    "mvlgamma",
    "mvlgamma_",
    "name",
    "names",
    "nanquantile",
    "nansum",
    "narrow",
    "narrow_copy",
    "ndim",
    "ndimension",
    "ne",
    "ne_",
    "neg",
    "neg_",
    "negative",
    "negative_",
    "nelement",
    "new",
    "new_empty",
    "new_full",
    "new_ones",
    "new_tensor",
    "new_zeros",
    "nextafter",
    "nextafter_",
    "nonzero",
    "norm",
    "normal_",
    "not_equal",
    "not_equal_",
    "numel",
    "numpy",
    "orgqr",
    "ormqr",
    "outer",
    "output_nr",
    "permute",
    "pin_memory",
    "pinverse",
    "polygamma",
    "polygamma_",
    "pow",
    "pow_",
    "prelu",
    "prod",
    "put_",
    "q_per_channel_axis",
    "q_per_channel_scales",
    "q_per_channel_zero_points",
    "q_scale",
    "q_zero_point",
    "qr",
    "qscheme",
    "quantile",
    "rad2deg",
    "rad2deg_",
    "random_",
    "real",
    "reciprocal",
    "reciprocal_",
    "record_stream",
    "refine_names",
    "register_hook",
    "reinforce",
    "relu",
    "relu_",
    "remainder",
    "remainder_",
    "rename",
    "rename_",
    "renorm",
    "renorm_",
    "repeat",
    "repeat_interleave",
    "requires_grad",
    "requires_grad_",
    "reshape",
    "reshape_as",
    "resize",
    "resize_",
    "resize_as",
    "resize_as_",
    "retain_grad",
    "rfft",
    "roll",
    "rot90",
    "round",
    "round_",
    "rsqrt",
    "rsqrt_",
    "scatter",
    "scatter_",
    "scatter_add",
    "scatter_add_",
    "select",
    "set_",
    "sgn",
    "sgn_",
    "shape",
    "share_memory_",
    "short",
    "sigmoid",
    "sigmoid_",
    "sign",
    "sign_",
    "signbit",
    "sin",
    "sin_",
    "sinh",
    "sinh_",
    "size",
    "slogdet",
    "smm",
    "softmax",
    "solve",
    "sort",
    "sparse_dim",
    "sparse_mask",
    "sparse_resize_",
    "sparse_resize_and_clear_",
    "split",
    "split_with_sizes",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "squeeze",
    "squeeze_",
    "sspaddmm",
    "std",
    "stft",
    "storage",
    "storage_offset",
    "storage_type",
    "stride",
    "sub",
    "sub_",
    "subtract",
    "subtract_",
    "sum",
    "sum_to_size",
    "svd",
    "symeig",
    "t",
    "t_",
    "take",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "to",
    "to_dense",
    "to_mkldnn",
    "to_sparse",
    "tolist",
    "topk",
    "trace",
    "transpose",
    "transpose_",
    "triangular_solve",
    "tril",
    "tril_",
    "triu",
    "triu_",
    "true_divide",
    "true_divide_",
    "trunc",
    "trunc_",
    "type",
    "type_as",
    "unbind",
    "unflatten",
    "unfold",
    "uniform_",
    "unique",
    "unique_consecutive",
    "unsafe_chunk",
    "unsafe_split",
    "unsafe_split_with_sizes",
    "unsqueeze",
    "unsqueeze_",
    "values",
    "var",
    "vdot",
    "view",
    "view_as",
    "where",
    "zero_",
]


### TODOS ###
"""
1)  Implement std and var functions
2)  Finish setting method for indexing
3)  Enforce type constraint that data is always some type of float, 
    scale is always some type of int
4)  Find elegant way of resolving issue with zero slices. These currently
    have an arbitrary scale tensor, which could lead to some issue later.
    Find a way of setting the scale values *really* small, something like
    the minimum value of the integer datatype
5)  Implement view, reshaping, and element setting operations on stensors
    with non-scalar scale tensors
6)  Implement device changing operations (to, cuda, cpu)
7)  Write custom version of matmul to give better data_dims checking, 
    given all the behaviors present in the original torch version

"""

GENERIC_ERROR = NotImplementedError(
    "Something went wrong, please let me "
    "know at https://github.com/jemisjoky/STensor/issues"
)

if __name__ == "__main__":
    matrix = torch.randn(5, 5)
    vector = torch.randn(5)
    vector = stensor(vector)
    print(matrix.mv(vector))
    print(torch.mv(matrix, vector).torch())

    # # Get the nontrivial attributes of a Pytorch tensor
    # class Objer:
    #     def __init__(self):
    #         pass
    # obj = Objer()
    # tens_atts = [f for f in dir(torch.ones(2)) if f not in dir(obj)]
    # stens_atts = [f for f in dir(stensor(torch.ones(2), (0,))) if f not in dir(obj)]
    # hom_atts = [f for f in tens_atts if f in HOMOG]
    # torch_atts = [f for f in tens_atts if f in TORCH]
    # other_atts = [f for f in tens_atts if f not in stens_atts]
    # print(other_atts)

    # # Make sure there aren't functions which can be overwridden but don't appear above
    # func_dict = torch._overrides.get_overridable_functions()
    # fun_names = [f.__name__ for f in func_dict[torch]]
    # for name in fun_names:
    #     if all(name not in big_set for big_set in [HOMOG, DOUBTFUL, TORCH, DO_NOW, DO_SOON, LATER]):
    #         assert False, name

    # import inspect
    # override_dict = torch._overrides.get_testing_overrides()
    # for fun_name in HOMOG:
    #     if HOMOG[fun_name][0][0] != () and HOMOG[fun_name][1][0] != ():
    #         continue
    #     dummy_fun = override_dict[getattr(torch, fun_name)]
    #     print(f"{fun_name}: {inspect.signature(dummy_fun)}")
