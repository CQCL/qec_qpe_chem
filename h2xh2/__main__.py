"""Command-line utility to drive the H2xH2 experiments."""

import json
import importlib
from argparse import ArgumentParser
from pytket.backends.backendresult import BackendResult
from ._cmdutils import (
    prepare,
    execute,
    retrieve,
)
from .experiment._chemistry import ChemData
from .experiment._benchmark import interpret_process_benchmark_results
from .experiment._iqpe import interpret_process_iqpe_results


parser = ArgumentParser(
    prog="h2xh2",
    description="H2xH2 command-line utilities",
)
parser.add_argument(
    "target",
    help="operation target [prepare|execute|retrieve|benchmark|iqpe|chem]",
)
parser.add_argument(
    "-i",
    dest="input_file",
    default="_input",
    help="Input file containing backend_input:BackendInput object (default: _input).",
)
parser.add_argument(
    "-o",
    dest="result_file",
    default="_backend_result.json",
    help="Output json file containing list[BackendResult] (default: _backend_results.json).",
)


def main():
    """Command-line utility driver."""
    args = parser.parse_args()
    m = importlib.import_module(args.input_file)
    backend_input = m.backend_input
    match args.target:
        case "prepare":
            prepare(backend_input)
        case "execute":
            execute(backend_input)
        case "retrieve":
            results = retrieve(backend_input)
            data = [r.to_dict() for r in results]
            with open(args.result_file, "w") as f:
                json.dump(data, f)
        case "benchmark":
            with open(args.result_file) as f:
                data = json.load(f)
            results = [BackendResult.from_dict(d) for d in data]
            val = interpret_process_benchmark_results(
                results, m.benchmark_input
            )._asdict()
            print(json.dumps(val))
        case "iqpe":
            with open(args.result_file) as f:
                data = json.load(f)
            results = [BackendResult.from_dict(d) for d in data]
            val = interpret_process_iqpe_results(results, m.iqpe_input)._asdict()
            print(json.dumps(val))
        case "chem":
            for k, v in ChemData()._asdict().items():
                print(k)
                print("    ", v)
        case _:
            raise KeyError(f"Unrecognized tareget: `{args.target}`")


if __name__ == "__main__":
    main()
