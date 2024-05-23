# -*- coding: utf-8 -*-

# xzl: from https://github.com/pytorch/tutorials/blob/main/recipes_source/recipes/amp_recipe.py

"""
Automatic Mixed Precision
*************************
**Author**: `Michael Carilli <https://github.com/mcarilli>`_

`torch.cuda.amp <https://pytorch.org/docs/stable/amp.html>`_ provides convenience methods for mixed precision,
where some operations use the ``torch.float32`` (``float``) datatype and other operations
use ``torch.float16`` (``half``). Some ops, like linear layers and convolutions,
are much faster in ``float16`` or ``bfloat16``. Other ops, like reductions, often require the dynamic
range of ``float32``.  Mixed precision tries to match each op to its appropriate datatype,
which can reduce your network's runtime and memory footprint.

Ordinarily, "automatic mixed precision training" uses `torch.autocast <https://pytorch.org/docs/stable/amp.html#torch.autocast>`_ and
`torch.cuda.amp.GradScaler <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler>`_ together.

This recipe measures the performance of a simple network in default precision,
then walks through adding ``autocast`` and ``GradScaler`` to run the same network in
mixed precision with improved performance.

You may download and run this recipe as a standalone Python script.
The only requirements are PyTorch 1.6 or later and a CUDA-capable GPU.

Mixed precision primarily benefits Tensor Core-enabled architectures (Volta, Turing, Ampere).
This recipe should show significant (2-3X) speedup on those architectures.
On earlier architectures (Kepler, Maxwell, Pascal), you may observe a modest speedup.
Run ``nvidia-smi`` to display your GPU's architecture.
"""

import torch, time, gc

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

##########################################################
# A simple network
# ----------------
# The following sequence of linear layers and ReLUs should show a speedup with mixed precision.

def make_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers)).cuda()

##########################################################
# ``batch_size``, ``in_size``, ``out_size``, and ``num_layers`` are chosen to be large enough to saturate the GPU with work.
# Typically, mixed precision provides the greatest speedup when the GPU is saturated.
# Small networks may be CPU bound, in which case mixed precision won't improve performance.
# Sizes are also chosen such that linear layers' participating dimensions are multiples of 8,
# to permit Tensor Core usage on Tensor Core-capable GPUs (see :ref:`Troubleshooting<troubleshooting>` below).
#
# Exercise: Vary participating sizes and see how the mixed precision speedup changes.

batch_size = 512 # Try, for example, 128, 256, 513.
in_size = 4096
out_size = 4096
num_layers = 3
num_batches = 50
epochs = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

# Creates data in default precision.
# The same data is used for both default and mixed precision trials below.
# You don't need to manually change inputs' ``dtype`` when enabling mixed precision.
data = [torch.randn(batch_size, in_size) for _ in range(num_batches)]
targets = [torch.randn(batch_size, out_size) for _ in range(num_batches)]

loss_fn = torch.nn.MSELoss().cuda()

##########################################################
# Default Precision
# -----------------
# Without ``torch.cuda.amp``, the following simple network executes all ops in default precision (``torch.float32``):

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        output = net(input)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
end_timer_and_print("Default precision:")

##########################################################
# Adding ``torch.autocast``
# -------------------------
# Instances of `torch.autocast <https://pytorch.org/docs/stable/amp.html#autocasting>`_
# serve as context managers that allow regions of your script to run in mixed precision.
#
# In these regions, CUDA ops run in a ``dtype`` chosen by ``autocast``
# to improve performance while maintaining accuracy.
# See the `Autocast Op Reference <https://pytorch.org/docs/stable/amp.html#autocast-op-reference>`_
# for details on what precision ``autocast`` chooses for each op, and under what circumstances.

for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        # Runs the forward pass under ``autocast``.
        # xzl: dtype only chooses between float16 or bf16
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            # output is float16 because linear layers ``autocast`` to float16.
            assert output.dtype is torch.float16

            loss = loss_fn(output, target)
            # loss is float32 because ``mse_loss`` layers ``autocast`` to float32.
            assert loss.dtype is torch.float32

        # Exits ``autocast`` before backward().
        # Backward passes under ``autocast`` are not recommended.
        # Backward ops run in the same ``dtype`` ``autocast`` chose for corresponding forward ops.
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance

##########################################################
# Adding ``GradScaler``
# ---------------------
# `Gradient scaling <https://pytorch.org/docs/stable/amp.html#gradient-scaling>`_
# helps prevent gradients with small magnitudes from flushing to zero
# ("underflowing") when training with mixed precision.
#
# `torch.cuda.amp.GradScaler <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler>`_
# performs the steps of gradient scaling conveniently.

# Constructs a ``scaler`` once, at the beginning of the convergence run, using default arguments.
# If your network fails to converge with default ``GradScaler`` arguments, please file an issue.
# The same ``GradScaler`` instance should be used for the entire convergence run.
# If you perform multiple convergence runs in the same script, each run should use
# a dedicated fresh ``GradScaler`` instance. ``GradScaler`` instances are lightweight.
scaler = torch.cuda.amp.GradScaler()

for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            loss = loss_fn(output, target)

        # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
        scaler.scale(loss).backward()

        # ``scaler.step()`` first unscales the gradients of the optimizer's assigned parameters.
        # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(opt)

        # Updates the scale for next iteration.
        scaler.update()

        opt.zero_grad() # set_to_none=True here can modestly improve performance

##########################################################
# All together: "Automatic Mixed Precision"
# ------------------------------------------
# (The following also demonstrates ``enabled``, an optional convenience argument to ``autocast`` and ``GradScaler``.
# If False, ``autocast`` and ``GradScaler``\ 's calls become no-ops.
# This allows switching between default precision and mixed precision without if/else statements.)

use_amp = True

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
end_timer_and_print("Mixed precision:")

##########################################################
# Inspecting/modifying gradients (e.g., clipping)
# --------------------------------------------------------
# All gradients produced by ``scaler.scale(loss).backward()`` are scaled.  If you wish to modify or inspect
# the parameters' ``.grad`` attributes between ``backward()`` and ``scaler.step(optimizer)``, you should
# unscale them first using `scaler.unscale_(optimizer) <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.unscale_>`_.

for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned parameters in-place
        scaler.unscale_(opt)

        # Since the gradients of optimizer's assigned parameters are now unscaled, clips as usual.
        # You may use the same value for max_norm here as you would without gradient scaling.
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)

        scaler.step(opt)
        scaler.update()
        opt.zero_grad() # set_to_none=True here can modestly improve performance

##########################################################
# Saving/Resuming
# ----------------
# To save/resume Amp-enabled runs with bitwise accuracy, use
# `scaler.state_dict <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.state_dict>`_ and
# `scaler.load_state_dict <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.load_state_dict>`_.
#
# When saving, save the ``scaler`` state dict alongside the usual model and optimizer state ``dicts``.
# Do this either at the beginning of an iteration before any forward passes, or at the end of
# an iteration after ``scaler.update()``.

checkpoint = {"model": net.state_dict(),
              "optimizer": opt.state_dict(),
              "scaler": scaler.state_dict()}
# Write checkpoint as desired, e.g.,
# torch.save(checkpoint, "filename")

##########################################################
# When resuming, load the ``scaler`` state dict alongside the model and optimizer state ``dicts``.
# Read checkpoint as desired, for example:
#
# .. code-block::
#
#    dev = torch.cuda.current_device()
#    checkpoint = torch.load("filename",
#                            map_location = lambda storage, loc: storage.cuda(dev))
#
net.load_state_dict(checkpoint["model"])
opt.load_state_dict(checkpoint["optimizer"])
scaler.load_state_dict(checkpoint["scaler"])

##########################################################
# If a checkpoint was created from a run *without* Amp, and you want to resume training *with* Amp,
# load model and optimizer states from the checkpoint as usual.  The checkpoint won't contain a saved ``scaler`` state, so
# use a fresh instance of ``GradScaler``.
#
# If a checkpoint was created from a run *with* Amp and you want to resume training *without* ``Amp``,
# load model and optimizer states from the checkpoint as usual, and ignore the saved ``scaler`` state.

##########################################################
# Inference/Evaluation
# --------------------
# ``autocast`` may be used by itself to wrap inference or evaluation forward passes. ``GradScaler`` is not necessary.

##########################################################
# .. _advanced-topics:
#
# Advanced topics
# ---------------
# See the `Automatic Mixed Precision Examples <https://pytorch.org/docs/stable/notes/amp_examples.html>`_ for advanced use cases including:
#
# * Gradient accumulation
# * Gradient penalty/double backward
# * Networks with multiple models, optimizers, or losses
# * Multiple GPUs (``torch.nn.DataParallel`` or ``torch.nn.parallel.DistributedDataParallel``)
# * Custom autograd functions (subclasses of ``torch.autograd.Function``)
#
# If you perform multiple convergence runs in the same script, each run should use
# a dedicated fresh ``GradScaler`` instance. ``GradScaler`` instances are lightweight.
#
# If you're registering a custom C++ op with the dispatcher, see the
# `autocast section <https://pytorch.org/tutorials/advanced/dispatcher.html#autocast>`_
# of the dispatcher tutorial.

##########################################################
# .. _troubleshooting:
#
# Troubleshooting
# ---------------
# Speedup with Amp is minor
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Your network may fail to saturate the GPU(s) with work, and is therefore CPU bound. Amp's effect on GPU performance
#    won't matter.
#
#    * A rough rule of thumb to saturate the GPU is to increase batch and/or network size(s)
#      as much as you can without running OOM.
#    * Try to avoid excessive CPU-GPU synchronization (``.item()`` calls, or printing values from CUDA tensors).
#    * Try to avoid sequences of many small CUDA ops (coalesce these into a few large CUDA ops if you can).
# 2. Your network may be GPU compute bound (lots of ``matmuls``/convolutions) but your GPU does not have Tensor Cores.
#    In this case a reduced speedup is expected.
# 3. The ``matmul`` dimensions are not Tensor Core-friendly.  Make sure ``matmuls`` participating sizes are multiples of 8.
#    (For NLP models with encoders/decoders, this can be subtle.  Also, convolutions used to have similar size constraints
#    for Tensor Core use, but for CuDNN versions 7.3 and later, no such constraints exist.  See
#    `here <https://github.com/NVIDIA/apex/issues/221#issuecomment-478084841>`_ for guidance.)
#
# Loss is inf/NaN
# ~~~~~~~~~~~~~~~
# First, check if your network fits an :ref:`advanced use case<advanced-topics>`.
# See also `Prefer binary_cross_entropy_with_logits over binary_cross_entropy <https://pytorch.org/docs/stable/amp.html#prefer-binary-cross-entropy-with-logits-over-binary-cross-entropy>`_.
#
# If you're confident your Amp usage is correct, you may need to file an issue, but before doing so, it's helpful to gather the following information:
#
# 1. Disable ``autocast`` or ``GradScaler`` individually (by passing ``enabled=False`` to their constructor) and see if ``infs``/``NaNs`` persist.
# 2. If you suspect part of your network (e.g., a complicated loss function) overflows , run that forward region in ``float32``
#    and see if ``infs``/``NaN``s persist.
#    `The autocast docstring <https://pytorch.org/docs/stable/amp.html#torch.autocast>`_'s last code snippet
#    shows forcing a subregion to run in ``float32`` (by locally disabling ``autocast`` and casting the subregion's inputs).
#
# Type mismatch error (may manifest as ``CUDNN_STATUS_BAD_PARAM``)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``Autocast`` tries to cover all ops that benefit from or require casting.
# `Ops that receive explicit coverage <https://pytorch.org/docs/stable/amp.html#autocast-op-reference>`_
# are chosen based on numerical properties, but also on experience.
# If you see a type mismatch error in an ``autocast`` enabled forward region or a backward pass following that region,
# it's possible ``autocast`` missed an op.
#
# Please file an issue with the error backtrace.  ``export TORCH_SHOW_CPP_STACKTRACES=1`` before running your script to provide
# fine-grained information on which backend op is failing.
