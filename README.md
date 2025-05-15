
H2xH2 project resources
=======================

This repository provides python tooling for running a Quantum Phase Estimation experiment for H2 on Quantinuum hardware presented in the paper:

K. Yamamoto et al., "Quantum Error-Corrected Computation of Molecular Energies", [arXiv:2505.09133](https://arxiv.org/abs/2505.09133)

It provides automatic tooling for running experiments on Quantinuum hardware using Steane code primitives, with support for QEC cycles and Iceberg error detection cycles. Rz gates are implemented with gate teleportation techniques. This repository is built on top of the [`pytket`](https://tket.quantinuum.com/api-docs/) library and contains all relevant source code.

`pytket` is a python module for interfacing with `TKET`, a quantum computing toolkit and optimising compiler developed by Quantinuum, and is available through `pip`. `pytket` is open source and documentation and use examples can be found at the [github repository](https://github.com/CQCL/pytket). <br>

## Disclaimer

We **stress** that the python tools available in this repository were written for the H2 on Quantinuum hardware experiment *add citation as appropriate*. We do not consider this code to be a prototype for running such experiments in the future. We are happy for code in this repository to be used in other work, but we do not guarantee that the code will work appropriately and any requires fixed will likely need to be made independently. While there are example notebooks and tests, API documentations and manuals are not provided.

## Installation

`h2xh2` package components can be installed directly from source via pip
```
pip install -e .
```

## Contact

Questions can be directed to <kentaro.yamamoto@quantinuum.com> or <silas.dilkes@quantinuum.com>.

