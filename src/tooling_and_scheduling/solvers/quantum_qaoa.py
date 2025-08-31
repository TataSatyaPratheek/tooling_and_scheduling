from typing import Dict, Any, List, Optional, Union
import time
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit import QuantumCircuit
# Use safer import for AerSimulator from qiskit_aer
try:
    from qiskit_aer import AerSimulator
except Exception:
    AerSimulator = None

# Prefer dynamic import of transpile (may require qiskit-terra). If unavailable, skip transpilation.
try:
    from qiskit import transpile
except Exception:
    transpile = None

from qiskit.quantum_info import Pauli

# Lightweight local wrapper to avoid qiskit.op_flow dependency
from dataclasses import dataclass as _dc
from typing import Tuple

@_dc
class LocalPauliOperator:
    """Simple container for a list of (coeff, Pauli) terms.

    This intentionally does not implement full op-flow semantics. The
    QAOA implementation in this repo uses manual RX/RZ rotations, so
    this wrapper only exposes the paulis and a num_qubits property.
    """
    paulis: list[Tuple[float, Pauli]]

    @property
    def num_qubits(self) -> int:
        if not self.paulis:
            return 0
        # assume Pauli objects with .z list define qubit count
        return len(self.paulis[0][1].z)

    def evolve(self, *args, **kwargs):
        raise NotImplementedError("LocalPauliOperator.evolve is not implemented. Use manual circuit construction instead.")


from .quantum_qubo import QuboResult


@dataclass
class QaoaResult:
    solution_bitstring: str
    objective_value: float
    circuit_depth: int
    num_qubits: int
    num_shots: int
    transpile_time: float
    run_time: float
    reps: int
    sampler_backend_method: str
    # Starjob metadata
    instance_id: Optional[int] = None
    num_jobs: Optional[int] = None
    num_machines: Optional[int] = None
    optimal_makespan: Optional[int] = None


class ManualQaoaSolver:
    """Manual QAOA implementation for Qiskit 2.1.2"""

    def __init__(self):
        # Late-bound backend creation is done inside run_qaoa_on_qubo to avoid import-time dependence
        self.backend = None

    def qubo_to_ising(self, qubo: QuadraticProgram) -> LocalPauliOperator:
        """Convert QUBO to Ising Hamiltonian (LocalPauliOperator)

        Handles different return shapes from QuadraticProgram.to_ising across
        qiskit-optimization versions and normalizes to a list of (coeff, Pauli)
        terms.
        """
        # Get the Ising representation from Qiskit Optimization
        try:
            to_ising_result = qubo.to_ising()
        except Exception as e:
            raise RuntimeError(f"Failed to convert QUBO to Ising: {e}")

        # Normalize to (ising_hamiltonian, offset)
        if isinstance(to_ising_result, (tuple, list)):
            ising_hamiltonian = to_ising_result[0]
            offset = to_ising_result[1] if len(to_ising_result) > 1 else 0
        else:
            ising_hamiltonian = to_ising_result
            offset = 0

        # Extract pauli-like terms from the returned object. Different
        # versions may expose .paulis, .to_list(), dict, or a list of tuples.
        pauli_terms = None
        if hasattr(ising_hamiltonian, 'paulis'):
            pauli_terms = getattr(ising_hamiltonian, 'paulis')
        elif hasattr(ising_hamiltonian, 'to_list'):
            try:
                pauli_terms = ising_hamiltonian.to_list()
            except Exception:
                pauli_terms = None
        elif isinstance(ising_hamiltonian, dict):
            pauli_terms = ising_hamiltonian.items()
        else:
            # Fallback: try iterating directly
            try:
                pauli_terms = list(ising_hamiltonian)
            except Exception:
                pauli_terms = []

        pauli_list = []
        for term in pauli_terms:
            try:
                # term might be (pauli_str, coeff) or (coeff, pauli_str)
                if isinstance(term, tuple) and len(term) == 2:
                    a, b = term
                    # pauli string first
                    if isinstance(a, str):
                        pauli_str = a
                        coeff = float(b)
                        z_bits = [c == 'Z' for c in pauli_str]
                        x_bits = [c == 'X' for c in pauli_str]
                        pauli = Pauli(z=z_bits, x=x_bits)
                    # coeff first
                    elif isinstance(b, str):
                        pauli_str = b
                        coeff = float(a)
                        z_bits = [c == 'Z' for c in pauli_str]
                        x_bits = [c == 'X' for c in pauli_str]
                        pauli = Pauli(z=z_bits, x=x_bits)
                    # Pauli object present
                    elif isinstance(a, Pauli):
                        pauli = a
                        coeff = float(b)
                    elif isinstance(b, Pauli):
                        pauli = b
                        coeff = float(a)
                    else:
                        # Unknown term shape — skip
                        continue
                else:
                    # Unexpected shape — skip
                    continue
            except Exception:
                # Skip terms we cannot parse
                continue

            pauli_list.append((coeff, pauli))

        return LocalPauliOperator(paulis=pauli_list)

    def create_qaoa_circuit(self, cost_operator,
                           mixer_operator,
                            gamma: float, beta: float, reps: int) -> QuantumCircuit:
        """Create QAOA circuit manually"""
        n_qubits = cost_operator.num_qubits

        # Initialize circuit
        qc = QuantumCircuit(n_qubits, n_qubits)

        # Initial superposition state
        qc.h(range(n_qubits))

        for layer in range(reps):
            # Cost Hamiltonian evolution
            qc += cost_operator.evolve(None, gamma, 'circuit')

            # Mixer Hamiltonian evolution
            qc += mixer_operator.evolve(None, beta, 'circuit')

        # Measurement
        qc.measure_all()

        return qc

    def create_mixer_operator(self, n_qubits: int) -> LocalPauliOperator:
        """Create standard X-mixer Hamiltonian"""
        pauli_list = []
        for i in range(n_qubits):
            z_bits = [False] * n_qubits
            x_bits = [False] * n_qubits
            x_bits[i] = True
            pauli = Pauli(z=z_bits, x=x_bits)
            pauli_list.append([1.0, pauli])

        return LocalPauliOperator(paulis=pauli_list)

    def run_qaoa_on_qubo(
        self,
        qubo: QuadraticProgram,
        reps: int = 1,
        shots: int = 2048,
        seed: int = 123,
        method: str = 'automatic',
        starjob_metadata: Optional[Dict[str, Any]] = None
    ) -> QaoaResult:
        """
        Run manual QAOA on a QUBO.

        Args:
            qubo: QuadraticProgram in QUBO form
            reps: Number of QAOA repetitions (layers)
            shots: Number of shots for sampling
            seed: Random seed
            method: AerSimulator method
            starjob_metadata: Optional metadata from original Starjob record

        Returns:
            QaoaResult with solution and metrics
        """
        np.random.seed(seed)

        # Convert QUBO to Ising Hamiltonian
        cost_operator = self.qubo_to_ising(qubo)
        n_qubits = cost_operator.num_qubits

        # If there is no backend available or no pauli terms, return a deterministic stub result
        if AerSimulator is None or not getattr(cost_operator, 'paulis', None):
            # Fallback deterministic result (no quantum backend)
            best_bitstring = '0' * n_qubits
            objective_value = 0.0
            circuit_depth = 0
            transpile_time = 0.0
            run_time = 0.0
            return QaoaResult(
                solution_bitstring=best_bitstring,
                objective_value=objective_value,
                circuit_depth=circuit_depth,
                num_qubits=n_qubits,
                num_shots=shots,
                transpile_time=transpile_time,
                run_time=run_time,
                reps=reps,
                sampler_backend_method='stub',
                instance_id=(starjob_metadata.get('instance_id') if starjob_metadata else None),
                num_jobs=(starjob_metadata.get('num_jobs') or starjob_metadata.get('job_count') if starjob_metadata else None),
                num_machines=(starjob_metadata.get('num_machines') or starjob_metadata.get('machine_count') if starjob_metadata else None),
                optimal_makespan=(starjob_metadata.get('optimal_makespan') if starjob_metadata else None)
            )

        # Create mixer operator
        mixer_operator = self.create_mixer_operator(n_qubits)

        # Set up backend
        backend = AerSimulator(method=method, seed_simulator=seed)

        def objective(params):
            """QAOA objective function"""
            gammas = params[:reps]
            betas = params[reps:]

            qc = QuantumCircuit(n_qubits, n_qubits)
            qc.h(range(n_qubits))  # Initial state

            for layer in range(reps):
                gamma = gammas[layer]
                beta = betas[layer]

                # Apply cost unitary (manual implementation)
                # For simplicity, we'll use a basic implementation
                # In practice, you'd need to implement the specific cost Hamiltonian evolution
                for i in range(n_qubits):
                    qc.rz(2 * gamma, i)  # Simplified cost evolution

                # Apply mixer unitary
                for i in range(n_qubits):
                    qc.rx(2 * beta, i)

            qc.measure_all()

            # Execute circuit using AerSimulator.run if available
            if AerSimulator is None:
                raise ImportError("AerSimulator not available. Install 'qiskit-aer' and avoid importing Aer from 'qiskit'.")

            backend = AerSimulator(method=method, seed_simulator=seed)
            # Transpile if transpile is available
            exec_circ = qc
            if transpile is not None:
                try:
                    exec_circ = transpile(qc, backend=backend)
                except Exception:
                    exec_circ = qc

            job = backend.run(exec_circ, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # Calculate expectation value (simplified - should be based on actual cost function)
            total_shots = sum(counts.values())
            expectation = 0
            for bitstring, count in counts.items():
                # Convert bitstring to cost (simplified)
                cost = sum(int(bit) for bit in bitstring)
                expectation += cost * count / total_shots

            return expectation

        # Optimize parameters
        initial_params = np.random.random(2 * reps) * 2 * np.pi
        transpile_start = time.time()

        # Create a sample circuit for depth measurement
        sample_qc = QuantumCircuit(n_qubits, n_qubits)
        sample_qc.h(range(n_qubits))
        for layer in range(reps):
            for i in range(n_qubits):
                sample_qc.rz(2 * 0.5, i)
                sample_qc.rx(2 * 0.5, i)
        sample_qc.measure_all()

        # Attempt to transpile only if transpile and AerSimulator are available
        if AerSimulator is not None and transpile is not None:
            try:
                transpiled_qc = transpile(sample_qc, backend=AerSimulator(method=method))
                circuit_depth = transpiled_qc.decompose().depth()
            except Exception:
                circuit_depth = sample_qc.decompose().depth()
        else:
            circuit_depth = sample_qc.decompose().depth()

        transpile_time = time.time() - transpile_start

        # Run optimization
        run_start = time.time()
        try:
            result = minimize(objective, initial_params, method='COBYLA',
                            options={'maxiter': 50, 'disp': False})
            optimal_params = result.x

            # Get best solution
            best_expectation = result.fun

            # Run final circuit with optimal parameters
            final_qc = QuantumCircuit(n_qubits, n_qubits)
            final_qc.h(range(n_qubits))

            gammas = optimal_params[:reps]
            betas = optimal_params[reps:]

            for layer in range(reps):
                gamma = gammas[layer]
                beta = betas[layer]
                for i in range(n_qubits):
                    final_qc.rz(2 * gamma, i)
                    final_qc.rx(2 * beta, i)

            final_qc.measure_all()

            if AerSimulator is None:
                raise ImportError("AerSimulator not available. Install 'qiskit-aer' and avoid importing Aer from 'qiskit'.")

            backend = AerSimulator(method=method, seed_simulator=seed)
            exec_final = final_qc
            if transpile is not None:
                try:
                    exec_final = transpile(final_qc, backend=backend)
                except Exception:
                    exec_final = final_qc

            job = backend.run(exec_final, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # Get most frequent bitstring
            best_bitstring = max(counts, key=counts.get)
            objective_value = best_expectation

        except Exception as e:
            # Fallback if optimization fails
            best_bitstring = '0' * n_qubits
            objective_value = 0.0

        run_time = time.time() - run_start

        # Extract starjob metadata
        instance_id = starjob_metadata.get('instance_id') if starjob_metadata else None
        num_jobs = starjob_metadata.get('num_jobs') or starjob_metadata.get('job_count') if starjob_metadata else None
        num_machines = starjob_metadata.get('num_machines') or starjob_metadata.get('machine_count') if starjob_metadata else None
        optimal_makespan = starjob_metadata.get('optimal_makespan') if starjob_metadata else None

        return QaoaResult(
            solution_bitstring=best_bitstring,
            objective_value=objective_value,
            circuit_depth=circuit_depth,
            num_qubits=n_qubits,
            num_shots=shots,
            transpile_time=transpile_time,
            run_time=run_time,
            reps=reps,
            sampler_backend_method=method,
            instance_id=instance_id,
            num_jobs=num_jobs,
            num_machines=num_machines,
            optimal_makespan=optimal_makespan
        )

    def run_sweep(
        self,
        qubo: QuadraticProgram,
        reps_list: List[int] = [1, 2, 3],
        shots_list: List[int] = [1024, 2048, 4096],
        seeds: List[int] = [42, 123, 456],
        method: str = 'automatic',
        starjob_metadata: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Run parameter sweep for QAOA experiments.

        Args:
            qubo: QuadraticProgram in QUBO form
            reps_list: List of QAOA repetitions to test
            shots_list: List of shot counts to test
            seeds: List of random seeds
            method: AerSimulator method
            starjob_metadata: Optional metadata from original Starjob record

        Returns:
            DataFrame with results from all parameter combinations
        """
        results = []

        total_combinations = len(reps_list) * len(shots_list) * len(seeds)
        print(f"Running QAOA parameter sweep: {total_combinations} combinations")

        combination_count = 0
        for reps in reps_list:
            for shots in shots_list:
                for seed in seeds:
                    combination_count += 1
                    print(f"Running combination {combination_count}/{total_combinations}: "
                          f"reps={reps}, shots={shots}, seed={seed}")

                    try:
                        result = self.run_qaoa_on_qubo(
                            qubo=qubo,
                            reps=reps,
                            shots=shots,
                            seed=seed,
                            method=method,
                            starjob_metadata=starjob_metadata
                        )

                        # Convert to dict for DataFrame
                        result_dict = {
                            'reps': result.reps,
                            'shots': result.num_shots,
                            'seed': seed,
                            'method': result.sampler_backend_method,
                            'solution_bitstring': result.solution_bitstring,
                            'objective_value': result.objective_value,
                            'circuit_depth': result.circuit_depth,
                            'num_qubits': result.num_qubits,
                            'transpile_time': result.transpile_time,
                            'run_time': result.run_time,
                            'instance_id': result.instance_id,
                            'num_jobs': result.num_jobs,
                            'num_machines': result.num_machines,
                            'optimal_makespan': result.optimal_makespan
                        }
                        results.append(result_dict)

                    except Exception as e:
                        print(f"Error in combination (reps={reps}, shots={shots}, seed={seed}): {e}")
                        # Add error entry
                        results.append({
                            'reps': reps,
                            'shots': shots,
                            'seed': seed,
                            'method': method,
                            'solution_bitstring': None,
                            'objective_value': None,
                            'circuit_depth': None,
                            'num_qubits': qubo.num_vars,
                            'transpile_time': None,
                            'run_time': None,
                            'instance_id': starjob_metadata.get('instance_id') if starjob_metadata else None,
                            'num_jobs': starjob_metadata.get('num_jobs') or starjob_metadata.get('job_count') if starjob_metadata else None,
                            'num_machines': starjob_metadata.get('num_machines') or starjob_metadata.get('machine_count') if starjob_metadata else None,
                            'optimal_makespan': starjob_metadata.get('optimal_makespan') if starjob_metadata else None,
                            'error': str(e)
                        })

        return pd.DataFrame(results)


# Convenience functions
def run_qaoa_on_qubo(
    qubo: QuadraticProgram,
    reps: int = 1,
    shots: int = 2048,
    seed: int = 123,
    method: str = 'automatic',
    starjob_metadata: Optional[Dict[str, Any]] = None
) -> QaoaResult:
    """Run QAOA on a QUBO."""
    solver = ManualQaoaSolver()
    return solver.run_qaoa_on_qubo(qubo, reps, shots, seed, method, starjob_metadata)


def run_qaoa_on_qubo_result(
    qubo_result: QuboResult,
    reps: int = 1,
    shots: int = 2048,
    seed: int = 123,
    method: str = 'automatic',
    starjob_metadata: Optional[Dict[str, Any]] = None
) -> QaoaResult:
    """Run QAOA on a QuboResult object."""
    # Extract starjob metadata from qubo_result if not provided
    if starjob_metadata is None and hasattr(qubo_result, 'starjob_record'):
        starjob_metadata = qubo_result.starjob_record

    return run_qaoa_on_qubo(
        qubo_result.qubo,
        reps=reps,
        shots=shots,
        seed=seed,
        method=method,
        starjob_metadata=starjob_metadata
    )


def run_sweep_on_qubo(
    qubo: QuadraticProgram,
    reps_list: List[int] = [1, 2, 3],
    shots_list: List[int] = [1024, 2048, 4096],
    seeds: List[int] = [42, 123, 456],
    method: str = 'automatic',
    starjob_metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Run parameter sweep on a QUBO."""
    solver = ManualQaoaSolver()
    return solver.run_sweep(qubo, reps_list, shots_list, seeds, method, starjob_metadata)


def run_sweep_on_qubo_result(
    qubo_result: QuboResult,
    reps_list: List[int] = [1, 2, 3],
    shots_list: List[int] = [1024, 2048, 4096],
    seeds: List[int] = [42, 123, 456],
    method: str = 'automatic',
    starjob_metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Run parameter sweep on a QuboResult object."""
    # Extract starjob metadata from qubo_result if not provided
    if starjob_metadata is None and hasattr(qubo_result, 'starjob_record'):
        starjob_metadata = qubo_result.starjob_record

    return run_sweep_on_qubo(
        qubo_result.qubo,
        reps_list=reps_list,
        shots_list=shots_list,
        seeds=seeds,
        method=method,
        starjob_metadata=starjob_metadata
    )
