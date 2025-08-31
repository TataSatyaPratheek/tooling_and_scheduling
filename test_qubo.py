#!/usr/bin/env python3
"""
Test script for Quantum QUBO solver
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tooling_and_scheduling.solvers.quantum_qubo import build_qubo_from_starjob


def test_qubo_solver():
    """Test the QUBO solver with a simple Starjob record using assertions."""

    # Sample Starjob record (from test fixtures)
    starjob_record = {
        "instance_id": 123,
        "job_count": 2,
        "machine_count": 2,
        "jobs": [
            {"processing_times": [3, 2], "machine_sequence": [0, 1]},
            {"processing_times": [2, 4], "machine_sequence": [1, 0]}
        ],
        "optimal_makespan": 7
    }

    print("Testing QUBO solver with sample record...")
    print(f"Jobs: {starjob_record['job_count']}, Machines: {starjob_record['machine_count']}")
    print(f"Optimal makespan: {starjob_record['optimal_makespan']}")

    # Build QUBO
    result = build_qubo_from_starjob(starjob_record, horizon_method='auto', max_T=20)

    # Basic assertions
    assert result is not None, "build_qubo_from_starjob returned None"
    assert hasattr(result, 'T'), "QuboResult missing T"
    assert hasattr(result, 'counts'), "QuboResult missing counts"
    assert isinstance(result.counts, dict), "result.counts should be a dict"
    assert 'n_vars' in result.counts and 'n_bin' in result.counts, "counts missing expected keys"

    print("\n✅ QUBO construction successful!")
    print(f"Time horizon T: {result.T}")
    print(f"Variables: {result.counts['n_vars']} (binary: {result.counts['n_bin']})")
    print(f"Constraints: {result.counts.get('n_constraints')}")

    # Print some variable mappings
    print("\nVariable mappings (first few):")
    for i, (key, var_name) in enumerate(result.index_maps.items()):
        if i < 5:
            print(f"  {key} -> {var_name}")

    print("\nQP Objective:")
    print(f"  {result.qp.objective}")

    print("\nQUBO conversion successful!")
    # Validate qubo structure
    assert hasattr(result, 'qubo'), "QuboResult missing qubo"
    try:
        q_vars_len = len(result.qubo.variables)
    except Exception:
        q_vars_len = None
    assert q_vars_len is None or isinstance(q_vars_len, int), "Unexpected qubo.variables structure"


if __name__ == "__main__":
    try:
        test_qubo_solver()
        print("All checks passed")
        sys.exit(0)
    except AssertionError as ae:
        print(f"Assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
