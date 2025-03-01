# Project 1

Name: Annabelle Huang

CNET ID: ahuang02

To build and run the four advection programs, use the command `make advection`.

### Serial Lax

- Let `nodes = 1` and `cpus-per-task = 1`. Use the command `srun ./advection_serial <N> 1 1 1` in the batchfile.

### Shared Memory Parallel Lax

- Let `nodes = 1` and `cpus-per-task = <NCORES>`. Use the command `srun ./advection_shared <N> 1 1 <NCORES>` in the batchfile.

### Distributed Memory Parallel Lax

- Let `nodes = <NNODES>` and `cpus-per-task = 1`. Use the command `mpirun ./advection_distributed <N> 1 1 1` in the batchfile.

### Hybrid Parallel Lax

- Let `nodes = <NNODES>`, `ntasks-per-node = 1`, and `cpus-per-task = <NCORES_PER_NODE>`. Use the command `mpirun ./advection_hybrid <N> 1 1 <NCORES_PER_NODE>` in the batchfile.
