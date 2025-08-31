from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import (
    IntegerToBinary,
    InequalityToEquality,
    LinearEqualityToPenalty,
    QuadraticProgramToQubo
)


@dataclass
class QuboResult:
    qp: QuadraticProgram
    qubo: QuadraticProgram
    index_maps: Dict[Tuple[int, int, int], str]  # (j, o, t) -> var_name
    T: int
    counts: Dict[str, int]  # n_vars, n_bin, etc.
    penalty_weights: Dict[str, float]


class QuantumQuboSolver:
    def __init__(self):
        self.integer_to_binary = IntegerToBinary()
        self.inequality_to_equality = InequalityToEquality()
        self.linear_to_penalty = LinearEqualityToPenalty()
        self.qp_to_qubo = QuadraticProgramToQubo()

    def build_qubo_from_starjob(
        self,
        starjob_record: Dict[str, Any],
        horizon_method: str = 'auto',
        max_T: int = 200
    ) -> QuboResult:
        """
        Build QuadraticProgram from Starjob record and convert to QUBO.

        Args:
            starjob_record: Dict with keys: instance_id, job_count/machine_count or num_jobs/num_machines,
                          jobs (list of dicts with processing_times, machine_sequence),
                          optimal_makespan
            horizon_method: 'sum_durations', 'bound2x_opt', or 'auto'
            max_T: Maximum time horizon to prevent memory issues

        Returns:
            QuboResult with QP, QUBO, and metadata
        """
        # Normalize field names (handle both formats)
        instance_id = starjob_record.get('instance_id', starjob_record.get('id', 0))
        num_jobs = starjob_record.get('job_count', starjob_record.get('num_jobs', 0))
        num_machines = starjob_record.get('machine_count', starjob_record.get('num_machines', 0))
        jobs = starjob_record.get('jobs', [])
        optimal_makespan = starjob_record.get('optimal_makespan')

        # If num_jobs is missing or inconsistent, infer from jobs list
        if not jobs:
            raise ValueError("starjob_record contains no 'jobs' list")

        if not isinstance(jobs, list):
            raise TypeError("starjob_record['jobs'] must be a list of job dicts")

        inferred_num_jobs = len(jobs)
        if not isinstance(num_jobs, int) or num_jobs <= 0:
            num_jobs = inferred_num_jobs
        elif num_jobs != inferred_num_jobs:
            # warn and prefer actual jobs length
            print(f"Warning: num_jobs value ({num_jobs}) does not match length of jobs list ({inferred_num_jobs}), using {inferred_num_jobs}")
            num_jobs = inferred_num_jobs

        # Infer num_machines if missing or zero: find max machine id in sequences
        if not isinstance(num_machines, int) or num_machines <= 0:
            max_machine = -1
            for job in jobs:
                seq = job.get('machine_sequence', [])
                if seq:
                    max_machine = max(max_machine, max(seq))
            if max_machine >= 0:
                num_machines = max_machine + 1
            else:
                raise ValueError("Could not infer num_machines from job machine_sequence data")

        # Calculate time horizon T
        T = self._calculate_time_horizon(jobs, optimal_makespan, horizon_method, max_T)

        # Create QuadraticProgram
        qp = QuadraticProgram(name=f"Starjob_{instance_id}")

        # Add binary variables x[j,o,t]
        index_maps = {}
        var_count = 0

        for j in range(num_jobs):
            for o in range(len(jobs[j]['processing_times'])):
                for t in range(T):
                    var_name = f"x_{j}_{o}_{t}"
                    qp.binary_var(name=var_name)
                    index_maps[(j, o, t)] = var_name
                    var_count += 1

        # Add integer variable for makespan
        qp.integer_var(name="C_max", lowerbound=0, upperbound=T)
        var_count += 1

        # Add constraints
        self._add_exactly_one_constraints(qp, index_maps, num_jobs, jobs, T)
        self._add_precedence_constraints(qp, index_maps, num_jobs, jobs, T)
        self._add_machine_capacity_constraints(qp, index_maps, num_machines, jobs, T)
        self._add_makespan_constraints(qp, index_maps, num_jobs, jobs, T)

        # Set objective: minimize C_max
        qp.minimize(linear={"C_max": 1})

        # Convert to QUBO
        qubo = self._convert_to_qubo(qp)

        # Collect metadata
        counts = {
            'n_vars': var_count,
            'n_bin': var_count - 1,  # C_max is integer
            'n_constraints': len(qp.linear_constraints) + len(qp.quadratic_constraints),
            'T': T
        }

        penalty_weights = {
            'linear_equality_penalty': getattr(self.linear_to_penalty, '_penalty', 1.0)
        }

        return QuboResult(
            qp=qp,
            qubo=qubo,
            index_maps=index_maps,
            T=T,
            counts=counts,
            penalty_weights=penalty_weights
        )

    def _calculate_time_horizon(
        self,
        jobs: List[Dict[str, Any]],
        optimal_makespan: Optional[int],
        horizon_method: str,
        max_T: int
    ) -> int:
        """Calculate time horizon T based on method."""
        # Calculate sum of all processing times
        total_duration = sum(
            sum(job['processing_times']) for job in jobs
        )

        if horizon_method == 'sum_durations':
            T = total_duration
        elif horizon_method == 'bound2x_opt' and optimal_makespan is not None:
            T = 2 * optimal_makespan
        elif horizon_method == 'auto':
            if optimal_makespan is not None:
                T = min(total_duration, 2 * optimal_makespan)
            else:
                T = total_duration
        else:
            raise ValueError(f"Unknown horizon_method: {horizon_method}")

        return min(T, max_T)

    def _add_exactly_one_constraints(
        self,
        qp: QuadraticProgram,
        index_maps: Dict[Tuple[int, int, int], str],
        num_jobs: int,
        jobs: List[Dict[str, Any]],
        T: int
    ):
        """Add exactly-one start time constraints for each operation."""
        for j in range(num_jobs):
            for o in range(len(jobs[j]['processing_times'])):
                # sum_t x[j,o,t] = 1
                linear_terms = {index_maps[(j, o, t)]: 1 for t in range(T)}
                qp.linear_constraint(
                    linear=linear_terms,
                    sense='==',
                    rhs=1,
                    name=f"exactly_one_{j}_{o}"
                )

    def _add_precedence_constraints(
        self,
        qp: QuadraticProgram,
        index_maps: Dict[Tuple[int, int, int], str],
        num_jobs: int,
        jobs: List[Dict[str, Any]],
        T: int
    ):
        """Add precedence constraints between operations in same job."""
        for j in range(num_jobs):
            ops = jobs[j]['processing_times']
            for o in range(1, len(ops)):
                # sum_t t*x[j,o,t] >= sum_t t*x[j,o-1,t] + p[j,o-1]
                p_prev = ops[o-1]

                # Left side: sum_t t*x[j,o,t]
                left_linear = {index_maps[(j, o, t)]: t for t in range(T)}

                # Right side: sum_t t*x[j,o-1,t] + p_prev
                right_linear = {index_maps[(j, o-1, t)]: t for t in range(T)}

                # Convert to: left - right >= p_prev
                constraint_linear = {}
                for var, coeff in left_linear.items():
                    constraint_linear[var] = coeff
                for var, coeff in right_linear.items():
                    constraint_linear[var] = constraint_linear.get(var, 0) - coeff

                qp.linear_constraint(
                    linear=constraint_linear,
                    sense='>=',
                    rhs=p_prev,
                    name=f"precedence_{j}_{o}"
                )

    def _add_machine_capacity_constraints(
        self,
        qp: QuadraticProgram,
        index_maps: Dict[Tuple[int, int, int], str],
        num_machines: int,
        jobs: List[Dict[str, Any]],
        T: int
    ):
        """Add machine capacity constraints."""
        # First, build operation-to-machine mapping
        op_machine_map = {}
        for j, job in enumerate(jobs):
            for o, machine in enumerate(job['machine_sequence']):
                op_machine_map[(j, o)] = machine

        for m in range(num_machines):
            for tau in range(T):
                # For each time tau, sum over operations on machine m
                # that overlap with [tau, tau+1)
                linear_terms = {}

                for (j, o), machine in op_machine_map.items():
                    if machine != m:
                        continue

                    p = jobs[j]['processing_times'][o]
                    # Operation o of job j overlaps with tau if there exists t
                    # such that t <= tau < t + p
                    for t in range(max(0, tau - p + 1), min(T, tau + 1)):
                        if index_maps.get((j, o, t)):
                            linear_terms[index_maps[(j, o, t)]] = 1

                if linear_terms:
                    qp.linear_constraint(
                        linear=linear_terms,
                        sense='<=',
                        rhs=1,
                        name=f"machine_{m}_time_{tau}"
                    )

    def _add_makespan_constraints(
        self,
        qp: QuadraticProgram,
        index_maps: Dict[Tuple[int, int, int], str],
        num_jobs: int,
        jobs: List[Dict[str, Any]],
        T: int
    ):
        """Add makespan constraints for terminal operations."""
        for j in range(num_jobs):
            o_last = len(jobs[j]['processing_times']) - 1
            p_last = jobs[j]['processing_times'][o_last]

            # sum_t (t + p_last) * x[j,o_last,t] <= C_max
            # Equivalent to: sum_t (t + p_last) * x[j,o_last,t] - C_max <= 0

            linear_terms = {index_maps[(j, o_last, t)]: t + p_last for t in range(T)}
            linear_terms["C_max"] = -1

            qp.linear_constraint(
                linear=linear_terms,
                sense='<=',
                rhs=0,
                name=f"makespan_{j}"
            )

    def _convert_to_qubo(self, qp: QuadraticProgram) -> QuadraticProgram:
        """Convert QuadraticProgram to QUBO using converters pipeline."""
        # Step 1: Convert integer variables to binary
        qp_binary = self.integer_to_binary.convert(qp)

        # Step 2: Convert inequalities to equalities (if any)
        qp_equality = self.inequality_to_equality.convert(qp_binary)

        # Step 3: Convert linear equalities to penalty terms
        qp_penalty = self.linear_to_penalty.convert(qp_equality)

        # Step 4: Convert to QUBO
        qubo = self.qp_to_qubo.convert(qp_penalty)

        return qubo


# Convenience function
def build_qubo_from_starjob(
    starjob_record: Dict[str, Any],
    horizon_method: str = 'auto',
    max_T: int = 200
) -> QuboResult:
    """Build QUBO from Starjob record."""
    solver = QuantumQuboSolver()
    return solver.build_qubo_from_starjob(starjob_record, horizon_method, max_T)
