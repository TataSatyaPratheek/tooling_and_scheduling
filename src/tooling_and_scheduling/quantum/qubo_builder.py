"""Wrapper module exposing QUBO builder at package level.

This allows scripts to import tooling_and_scheduling.quantum.qubo_builder.build_qubo_from_starjob
while the implementation lives in tooling_and_scheduling.solvers.quantum_qubo.
"""
from typing import Dict, Any

try:
    from tooling_and_scheduling.solvers.quantum_qubo import build_qubo_from_starjob as build_qubo_from_starjob_impl
except Exception as e:
    build_qubo_from_starjob_impl = None


def build_qubo_from_starjob(starjob_record: Dict[str, Any], horizon_method: str = 'auto', max_T: int = 200):
    """Proxy function to the actual builder implementation.

    Raises ImportError if implementation not available.
    """
    if build_qubo_from_starjob_impl is None:
        raise ImportError("build_qubo_from_starjob implementation not found in solvers package")
    return build_qubo_from_starjob_impl(starjob_record, horizon_method=horizon_method, max_T=max_T)


__all__ = ["build_qubo_from_starjob"]
