import os
import sys
import time
import types
import pytest
from types import SimpleNamespace

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Use a tiny synthetic Starjob record to avoid dataset downloads
def make_tiny_starjob():
    return {
        'instance_id': 999,
        'num_jobs': 2,
        'num_machines': 2,
        'jobs': [
            {'processing_times': [1, 2], 'machine_sequence': [0, 1]},
            {'processing_times': [2, 1], 'machine_sequence': [1, 0]}
        ],
        'optimal_makespan': 4
    }


def test_t3_end_to_end_local(tmp_path):
    """End-to-end T-3 on a synthetic tiny record. Do not require Starjob dataset or Aer.

    - Build QUBO using the repo builder
    - Run a smoke QAOA using the real runner if available, else a lightweight stub
    - Persist metrics to CSV (using repo logger if present, else pandas)
    """
    # Import builder
    try:
        from tooling_and_scheduling.solvers.quantum_qubo import build_qubo_from_starjob
    except Exception as e:
        pytest.fail(f"QUBO builder not importable: {e}")

    # Try to import actual QAOA runner; if unavailable provide a deterministic stub
    try:
        from tooling_and_scheduling.solvers.quantum_qaoa import run_qaoa_on_qubo
        real_runner = True
    except Exception:
        real_runner = False

        def run_qaoa_on_qubo(qubo, reps=1, shots=128, seed=123, method='automatic', starjob_metadata=None):
            # Lightweight deterministic stub that simulates runtime and returns expected attributes
            start = time.time()
            time.sleep(0.01)  # simulate small runtime
            run_time = time.time() - start
            return SimpleNamespace(
                solution_bitstring='0' * 1,
                objective_value=1.0,
                circuit_depth=1,
                num_qubits=1,
                num_shots=shots,
                transpile_time=0.0,
                run_time=run_time,
                reps=reps,
                sampler_backend_method=method,
                instance_id=(starjob_metadata.get('instance_id') if starjob_metadata else None),
                num_jobs=(starjob_metadata.get('num_jobs') if starjob_metadata else None),
                num_machines=(starjob_metadata.get('num_machines') if starjob_metadata else None),
                optimal_makespan=(starjob_metadata.get('optimal_makespan') if starjob_metadata else None)
            )

    # Try to import metrics logger, but fall back to pandas if missing
    try:
        from tooling_and_scheduling.quantum.metrics import log_qaoa_results
        metrics_logger = True
    except Exception:
        metrics_logger = False
        import pandas as pd

    # Build QUBO from synthetic record
    record = make_tiny_starjob()
    qubo_result = build_qubo_from_starjob(record, horizon_method='auto', max_T=20)
    assert qubo_result is not None

    qubo = getattr(qubo_result, 'qubo', qubo_result)

    # Run QAOA (use small shots for speed)
    qres = run_qaoa_on_qubo(qubo, reps=1, shots=128, seed=42, method='automatic', starjob_metadata=record)

    # Basic checks
    for attr in ['objective_value', 'circuit_depth', 'num_qubits', 'run_time']:
        assert hasattr(qres, attr), f"Missing attribute {attr} in QAOA result"

    # Persist metrics
    row = vars(qres) if hasattr(qres, '__dict__') else dict(qres)
    if 'instance_id' not in row:
        row['instance_id'] = record.get('instance_id')

    out_csv = tmp_path / 'qaoa_results.csv'
    if metrics_logger:
        # Use repo logger
        log_qaoa_results([row], out_csv=str(out_csv), append=False)
        import pandas as pd
        df = pd.read_csv(out_csv)
    else:
        # Minimal persistence via pandas
        import pandas as pd
        df = pd.DataFrame([row])
        df.to_csv(out_csv, index=False)

    # Verify persisted columns
    expected_cols = ['instance_id', 'reps', 'shots', 'circuit_depth', 'objective_value', 'run_time']
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in persisted metrics"

    assert len(df) >= 1
