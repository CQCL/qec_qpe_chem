"""Command-line utils to drive experiments through Quantinuum Nexus (QNexus).

Thi experiments provides __main__.py to perform:

```sh
$ python -m h2xh2 _input prep       # Prepare the circuits, upload them to QNexus and compile there.
$ python -m h2xh2 _input exec       # Execute the circuits through QNexus.
$ python -m h2xh2 _input retr       # Retrieve the backend results from QNexus.
```

These commandline workflow facilitates performing multiple experiments.
One can entirely bypass this workflow to focus on a single experiments using Jupyter notebook.
"""

from typing import (
    NamedTuple,
    Callable,
    Mapping,
    Any,
)
import qnexus as qnx
from qnexus import QuantinuumConfig
from qnexus.models.references import ProjectRef
from pytket.circuit import Circuit
from qnexus.models.language import Language
from qnexus.models.filters import SortFilterEnum
from qnexus.exceptions import ZeroMatches
from pytket.backends.backendresult import BackendResult


def project_get(name: str, **kwargs: Mapping[str, Any]) -> ProjectRef:
    ls = qnx.projects.get_all(
        name_like=name,
        **kwargs,
    )
    project_ref_from_exact_name: ProjectRef | None = None
    for project_ref in ls:
        if name == project_ref.annotations.name:
            project_ref_from_exact_name = project_ref
    if project_ref_from_exact_name is None:
        raise KeyError(f"No exact matching with {name}")
    return project_ref_from_exact_name


class BackendInput(NamedTuple):
    """Utility class for preparing the input data.

    Args:
        project_name: Project name used by QNexus.
        backend_config: Backend config used by QNexus.
        get_circuits: Function to generate a list of (physical) circuits to be executed.
        language: Intermediate language (default: QASM).
        optimisation_level: Optimisation level used by the TKET compiler.
    """

    project_name: str
    backend_config: QuantinuumConfig
    get_circuits: Callable[[], list[Circuit]]
    get_n_shots: Callable[[], list[int]]
    language: Language = Language.QASM
    optimisation_level: int = 0


def prepare(
    backend_input: BackendInput,
) -> None:
    """Drive the preparation workflow (i.e., generate, upload, compile the circuits).

    Args:
        backend_input: BackendInput object.
    """
    # Create a project (must be `create``).
    my_project_ref = qnx.projects.create(
        name=backend_input.project_name,
    )
    # Upload the circuits.
    circuits = backend_input.get_circuits()
    for i, circ in enumerate(circuits):
        qnx.circuits.upload(
            name=f"encoded:{i:04d}",
            circuit=circ,
            project=my_project_ref,
        )
    my_circuit_refs = qnx.circuits.get_all(
        name_like="encoded",
        project=my_project_ref,
        sort_filters=[SortFilterEnum.NAME_ASC],
    ).list()
    # Compile the circuits.
    compiled_circuits = qnx.start_compile_job(
        circuits=my_circuit_refs,
        name="comp",
        optimisation_level=backend_input.optimisation_level,
        backend_config=backend_input.backend_config,
        project=my_project_ref,
    )
    compiled_circuits.df()


def execute(
    backend_input: BackendInput,
) -> None:
    """Drive the execution workflow.

    Args:
        backend_input: BackendInput object.
    """
    # Get the existing project.
    # HACK: Apparently projects.get() does not allow the exact project name.
    # my_project_ref = qnx.projects.get(
    #     name_like=backend_input.project_name,
    # )
    my_project_ref = project_get(name=backend_input.project_name)
    # Check if the circuits are already executed.
    execute_job_ref = qnx.jobs.get_all(
        name_like="exec",
        project=my_project_ref,
    )
    try:
        execute_job_ref.df()
    except ZeroMatches:
        pass
    else:
        print("Job already submitted")
        exit()
    # Get the final compilation circuit.
    compiled_circuits = qnx.circuits.get_all(
        name_like="-final",
        project=my_project_ref,
        sort_filters=[SortFilterEnum.NAME_ASC],
    ).list()
    # Submit the execution jobs.
    n_shots = backend_input.get_n_shots()
    qnx.start_execute_job(
        circuits=compiled_circuits,
        name="exec",
        n_shots=n_shots,
        backend_config=backend_input.backend_config,
        project=my_project_ref,
        language=backend_input.language,
    )


def retrieve(
    backend_input: BackendInput,
) -> list[BackendResult]:
    """Drive the retrieve workflow to download a list of BackendResult.

    Args:
        backend_input: BackendInput object.

    Returns:
        Backend results.
    """
    # Get the project reference.
    print(backend_input.project_name)
    # HACK: Apparently projects.get() does not allow the exact project name.
    # my_project_ref = qnx.projects.get(
    #     name_like=backend_input.project_name,
    # )
    my_project_ref = project_get(name=backend_input.project_name)
    # Get the backend results.
    execute_job_ref = qnx.jobs.get_all(
        name_like="exec",
        project=my_project_ref,
        sort_filters=[SortFilterEnum.NAME_ASC],
    )
    try:
        execute_job_ref.df()
    except ZeroMatches:
        print("Error: no job is found")
        exit()
    execute_job_ref = execute_job_ref.list()[0]
    # Check if the result is available.
    qnx.jobs.wait_for(execute_job_ref, timeout=5)
    # Retrieve a ExecutionResultRef for every Circuit that was executed.
    execute_job_result_refs = qnx.jobs.results(execute_job_ref)
    # Get a pytket BackendResult for the execution
    # for ref in execute_job_result_refs:
    #     print(ref.df())
    results = [rref.download_result() for rref in execute_job_result_refs]
    return results
