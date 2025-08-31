# src/tooling_and_scheduling/models/job_shop.py
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum


class Operation(BaseModel):
    """Single operation within a job"""
    job_id: int
    operation_id: int
    machine_id: int
    duration: int
    position_in_job: int


class Job(BaseModel):
    """Complete job with ordered operations"""
    job_id: int
    operations: List[Operation]
    total_duration: int


class JobShopInstance(BaseModel):
    """Complete job shop instance"""
    name: str
    num_jobs: int
    num_machines: int
    jobs: List[Job]
    operations: List[Operation]
    makespan_lower_bound: Optional[int] = None
    best_known_makespan: Optional[int] = None


class SolverType(str, Enum):
    DISPATCHING_RULE = "dispatching_rule"
    CP_SAT = "cp_sat"
    QAOA = "qaoa"


class SolverParams(BaseModel):
    """Parameters for different solvers"""
    solver_type: SolverType
    seed: int = Field(default=42, description="Random seed for reproducibility")
    # QAOA specific
    reps: Optional[int] = Field(default=1, ge=1, le=3, description="QAOA repetitions p")
    shots: Optional[int] = Field(default=1024, ge=1, description="Number of quantum shots")
    # Time limits
    time_limit_seconds: Optional[int] = Field(default=300, description="Solver time limit")


class Schedule(BaseModel):
    """Solution schedule"""
    operations: List[Dict[str, int]]  # {job_id, op_id, machine_id, start, end}
    makespan: int
    is_feasible: bool
    solver_type: SolverType


class Metrics(BaseModel):
    """Execution and quality metrics"""
    makespan: int
    runtime_seconds: float
    approximation_ratio: Optional[float] = None
    seed: int
    solver_type: SolverType
    # QAOA specific
    circuit_depth: Optional[int] = None
    num_shots: Optional[int] = None
    objective_value: Optional[float] = None
