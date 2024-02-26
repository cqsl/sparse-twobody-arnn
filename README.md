# Sparse Autoregressive Neural Networks for Classical Spin Systems

TODO: Paper link

The code uses [Equinox](https://github.com/patrick-kidger/equinox) to define neural networks.

`main.py` runs the VMC training of TwoBo or MADE, or the exact enumeration, given a Hamiltonian instance. `args.py` contains all the configurations.

`reproduce.sh` contains the commands to reproduce the TwoBo and MADE results in the paper. In practice, you may run these commands in parallel on multiple GPUs, and set your output directory.
