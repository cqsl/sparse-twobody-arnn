#!/bin/sh

# Figs. 2 and 3
for net in twobo made; do
    for L in 16 24 32; do
        for seed in $(seq 0 9); do
            echo main.py --ham_path ./instances/2D_EA_PM/2D_EA_PM_L"$L"_"$seed".npz --net_type $net ">" ./out/2D_EA_PM/2D_EA_PM_L"$L"_"$seed"_"$net".log
        done
    done
    for L in 4 8 12; do
        for seed in $(seq 0 9); do
            echo main.py --ham_path ./instances/3D_EA_PM/3D_EA_PM_L"$L"_"$seed".npz --net_type $net ">" ./out/3D_EA_PM/3D_EA_PM_L"$L"_"$seed"_"$net".log
        done
    done
    for N in 256 512 1024; do
        for seed in $(seq 0 9); do
            echo main.py --ham_path ./instances/RRG_PM/RRG_PM_d3_N"$N"_"$seed".npz --net_type $net ">" ./out/RRG_PM/RRG_PM_d3_N"$N"_"$seed"_"$net".log
        done
    done
done

# Fig. 4
for net in twobo made; do
    for opt_steps in 4 8 16 32 64 128 256 512 1024; do
        for seed in $(seq 0 9); do
            echo main.py --ham_path ./instances/2D_EA_PM/2D_EA_PM_L24_"$seed".npz --net_type $net --opt_steps $opt_steps ">" ./out/2D_EA_PM/2D_EA_PM_L24_"$seed"_"$net"_os"$opt_steps".log
        done
    done
done

# Fig. S2
for seed in $(seq 0 9); do
    echo main.py --ham_path ./instances/2D_EA_PM/2D_EA_PM_L32_"$seed".npz --net_type twobo --use_beta_skip 0 ">" ./out/2D_EA_PM/2D_EA_PM_L32_"$seed"_twobo_nobs.log
done
