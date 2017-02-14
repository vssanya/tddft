#!/bin/sh
#PBS -N jrcd_phase
#PBS -M vssanya@yandex.ru
#PBS -m abe
#PBS -o jrcd_phase_out.log
#PBS -e jrcd_phase_err.log
#PBS -l nodes=10:ppn=20
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR
mpirun python jrcd.py
