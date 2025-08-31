"""Wrapper module exposing QAOA runner at package level.

Proxies to tooling_and_scheduling.solvers.quantum_qaoa.run_qaoa_on_qubo using dynamic import
so that importing this module does not fail in environments without Qiskit.
"""
from typing import Dict, Any


def run_qaoa_on_qubo(qubo, reps: int = 1, shots: int = 1024, seed: int = 123, method: str = 'automatic', starjob_metadata: Dict[str, Any] = None):
    """Dynamically import and invoke the real QAOA runner implementation.

    Raises ImportError with a helpful message if the implementation or its dependencies are unavailable.
    """
    try:
        from tooling_and_scheduling.solvers.quantum_qaoa import run_qaoa_on_qubo as _impl
    except Exception as e:
        raise ImportError(
            "run_qaoa_on_qubo implementation not available. Ensure 'qiskit' and 'qiskit-aer' are installed "
            "and the solvers.quantum_qaoa module can be imported. Original error: {}".format(e)
        )

    return _impl(qubo, reps=reps, shots=shots, seed=seed, method=method, starjob_metadata=starjob_metadata)


__all__ = ["run_qaoa_on_qubo"]
