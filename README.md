# Sparse Autoregressive Neural Networks for Classical Spin Systems

Paper link: [arXiv:2402.16579](https://arxiv.org/abs/2402.16579) | [Mach. Learn.: Sci. Technol. 5 025074 (2024)](https://iopscience.iop.org/article/10.1088/2632-2153/ad5783)

The code uses [Equinox](https://github.com/patrick-kidger/equinox) to define neural networks.

`main.py` runs the VMC training of TwoBo or MADE, or the exact enumeration, given a Hamiltonian instance. `args.py` contains all the configurations.

`reproduce.sh` contains the commands to reproduce the TwoBo and MADE results in the paper. In practice, you may run these commands in parallel on multiple GPUs, and set your output directory.
