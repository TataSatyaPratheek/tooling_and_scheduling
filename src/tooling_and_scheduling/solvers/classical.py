# src/tooling_and_scheduling/solvers/classical.py
import time
from typing import Dict, List
from job_shop_lib import JobShopInstance
from job_shop_lib.dispatching import Dispatcher, ready_operations_filter_factory
from ortools.sat.python import cp_model

from ..models.job_shop import JobShopInstance, Schedule, Metrics, SolverParams, SolverType


class ClassicalSolver:
    """Classical baseline solvers for job shop scheduling"""
    
    def solve_with_dispatching_rule(
        self, 
        instance: JobShopInstance, 
        params: SolverParams,
        rule: str = "SPT"
    ) -> tuple[Schedule, Metrics]:
        """Solve using dispatching rules via JobShopLib"""
        
        # Convert to JobShopLib instance
        durations_matrix = []
        machines_matrix = []
        for job in instance.jobs:
            job_durations = []
            job_machines = []
            for op in sorted(job.operations, key=lambda x: x.position_in_job):
                job_durations.append(op.duration)
                job_machines.append(op.machine_id)
            durations_matrix.append(job_durations)
            machines_matrix.append(job_machines)
        
        jslib_instance = JobShopInstance.from_matrices(
            durations_matrix=durations_matrix,
            machines_matrix=machines_matrix,
            name=instance.name
        )
        
        # Select dispatching rule
        rule_mapping = {
            "SPT": "shortest_processing_time",  # Placeholder - need to check actual filter names
            "LPT": "longest_processing_time",   # Placeholder
        }
        
        if rule not in rule_mapping:
            raise ValueError(f"Unknown rule: {rule}. Available: {list(rule_mapping.keys())}")
        
        # For now, use a basic dispatcher without specific rules
        dispatcher = Dispatcher(jslib_instance)
        
        start_time = time.time()
        solution = dispatcher.solve()
        runtime = time.time() - start_time
        
        # Convert solution to our format
        schedule_ops = []
        for job_id, job_schedule in enumerate(solution.schedule):
            for op_pos, (machine_id, start_time, end_time) in enumerate(job_schedule):
                schedule_ops.append({
                    'job_id': job_id,
                    'op_id': job_id * len(job_schedule) + op_pos,
                    'machine_id': machine_id,
                    'start': start_time,
                    'end': end_time
                })
        
        schedule = Schedule(
            operations=schedule_ops,
            makespan=solution.makespan(),
            is_feasible=True,
            solver_type=SolverType.DISPATCHING_RULE
        )
        
        # Calculate approximation ratio if best known is available
        approx_ratio = None
        if instance.best_known_makespan:
            approx_ratio = solution.makespan() / instance.best_known_makespan
        
        metrics = Metrics(
            makespan=solution.makespan(),
            runtime_seconds=runtime,
            approximation_ratio=approx_ratio,
            seed=params.seed,
            solver_type=SolverType.DISPATCHING_RULE
        )
        
        return schedule, metrics
