import argparse

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument(
    "--ham_path",
    type=str,
    default="./instances/2D_EA_PM/2D_EA_PM_L4_0.npz",
    help="path to the Hamiltonian instance",
)

parser.add_argument(
    "--net_type",
    type=str,
    default="twobo",
    choices=["twobo", "made", "exact"],
    help="network type",
)
parser.add_argument(
    "--weight_skip",
    type=int,
    default=2,
    help="weight of the skip connection for twobo",
)
parser.add_argument(
    "--use_beta_skip",
    type=int,
    default=1,
    help="whether anneal the beta in the skip connection for twobo",
)

parser.add_argument(
    "--beta_range",
    type=str,
    default="0.05,3,60",
    help="values of beta in start,stop,num",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=500,
    help="optimization steps before starting the first annealing step",
)
parser.add_argument(
    "--opt_steps",
    type=int,
    default=200,
    help="optimization steps in each annealing step",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1024,
    help="batch size",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="learning rate",
)
parser.add_argument(
    "--opt_seed",
    type=int,
    default=0,
    help="random seed for optimization",
)

args = parser.parse_args()
args.use_beta_skip = bool(args.use_beta_skip)
