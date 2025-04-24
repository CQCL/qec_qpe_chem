
H2xH2 project resources
=======================

`h2xh2` stores python tooling for running a Quantum Phase Estimation experiment for H2 on Quantinuum hardware. It provides automatic tooling for running experiments on Quantinuum hardware with Rz gates are compiled with the gate teleportation technique and encoding gates with Steane code primitives. It is partially built on top of the [`pytket`](https://tket.quantinuum.com/api-docs/) library and this repository contains all of the relevant source code.

## Disclaimer

We **stress** that the python tools available in this repository were written for the H2 on Quantinuum hardware experiment *add citation as appropriate*. We do not consider this code as a prototype for running such experiments in the future.  If you are considering using code from this repository in your own work then we are very happy for you to do so, but we don't guarantee that the code will work suitably and likely any fixes will need to be made yourself.


## Installation

`h2xh2` package components can be installed directly from source via pip
```
pip install -e .
```

## Contact

Questions can be directed to <kentaro.yamamoto@quantinuum.com> or <silas.dilkes@quantinuum.com>.

