# src/tooling_and_scheduling/parsers/job_shop_parser.py
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from job_shop_lib import JobShopInstance
from job_shop_lib.benchmarking import load_benchmark_instance
from job_shop_lib.dispatching import (
    Dispatcher,
    ready_operations_filter_factory,
)

from ..models.job_shop import JobShopInstance, Operation, Job


class JobShopParser:
    """Parser for job shop instances using JobShopLib"""
    
    def __init__(self):
        self.available_instances = self._list_builtin_instances()
    
    def _list_builtin_instances(self) -> List[str]:
        """Get list of available built-in instances"""
        try:
            # Common benchmark instances available in job-shop-lib
            return [
                "ft06", "ft10", "ft20",  # Fisher and Thompson instances
                "la01", "la02", "la03", "la04", "la05",  # Lawrence instances
                "abz5", "abz6", "abz7", "abz8", "abz9",  # Adams, Balas, and Zawack
                "ta01", "ta02", "ta03", "ta04", "ta05",  # Taillard instances
            ]
        except Exception:
            return ["ft06", "ft10"]  # Fallback minimal set
    
    def load_instance(self, instance_name: str) -> JobShopInstance:
        """Load a built-in instance by name"""
        if instance_name not in self.available_instances:
            raise ValueError(f"Instance {instance_name} not available. Use: {self.available_instances}")
        
        try:
            # Load using JobShopLib benchmarking
            instance = load_benchmark_instance(instance_name)
            return self._convert_to_model(instance, instance_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load instance {instance_name}: {e}")
    
    def _convert_to_model(self, instance: JobShopInstance, name: str) -> JobShopInstance:
        """Convert JobShopLib Instance to our Pydantic model"""
        operations = []
        jobs = []
        
        for job_id, job_ops in enumerate(instance.jobs):
            job_operations = []
            for op in job_ops:
                operation = Operation(
                    job_id=op.job_id,
                    operation_id=op.operation_id,
                    machine_id=op.machine_id,
                    duration=op.duration,
                    position_in_job=op.position_in_job
                )
                operations.append(operation)
                job_operations.append(operation)
            
            job = Job(
                job_id=job_id,
                operations=job_operations,
                total_duration=sum(op.duration for op in job_operations)
            )
            jobs.append(job)
        
        return JobShopInstance(
            name=name,
            num_jobs=instance.num_jobs,
            num_machines=instance.num_machines,
            jobs=jobs,
            operations=operations,
            best_known_makespan=getattr(instance, 'best_known_makespan', None)
        )
    
    def export_to_json(self, instance: JobShopInstance, output_path: Path) -> None:
        """Export instance to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(instance.model_dump(), f, indent=2)
    
    def export_to_csv(self, instance: JobShopInstance, output_dir: Path) -> None:
        """Export instance to CSV files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Operations CSV
        ops_data = []
        for op in instance.operations:
            ops_data.append({
                'job_id': op.job_id,
                'operation_id': op.operation_id,
                'machine_id': op.machine_id,
                'duration': op.duration,
                'position_in_job': op.position_in_job
            })
        
        ops_df = pd.DataFrame(ops_data)
        ops_df.to_csv(output_dir / f"{instance.name}_operations.csv", index=False)
        
        # Instance metadata CSV
        metadata = {
            'name': instance.name,
            'num_jobs': instance.num_jobs,
            'num_machines': instance.num_machines,
            'total_operations': len(instance.operations),
            'best_known_makespan': instance.best_known_makespan
        }
        
        pd.DataFrame([metadata]).to_csv(
            output_dir / f"{instance.name}_metadata.csv", index=False
        )
    
    def validate_instance(self, instance: JobShopInstance) -> Dict[str, bool]:
        """Validate instance integrity"""
        checks = {}
        
        # Check job count consistency
        checks['job_count_consistent'] = len(instance.jobs) == instance.num_jobs
        
        # Check operations per job
        for job in instance.jobs:
            expected_ops = len(job.operations)
            actual_ops = len([op for op in instance.operations if op.job_id == job.job_id])
            checks[f'job_{job.job_id}_operation_count'] = expected_ops == actual_ops
        
        # Check machine IDs are valid
        valid_machines = set(range(instance.num_machines))
        used_machines = {op.machine_id for op in instance.operations}
        checks['machine_ids_valid'] = used_machines.issubset(valid_machines)
        
        # Check precedence ordering
        for job in instance.jobs:
            positions = [op.position_in_job for op in job.operations]
            checks[f'job_{job.job_id}_precedence_order'] = positions == sorted(positions)
        
        return checks
