# scripts/quick_start.py
#!/usr/bin/env python3
"""Quick start script for rapid prototyping"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tooling_and_scheduling.data_loader import DataLoader
from tooling_and_scheduling.solvers.classical import ClassicalSolver
from tooling_and_scheduling.models.job_shop import SolverParams, SolverType


def main():
    """Run a quick prototype test"""
    print("=== Job Shop Scheduling Rapid Prototype ===")
    
    # Load data
    loader = DataLoader()
    print("\n1. Loading small instances...")
    instances = loader.load_small_instances()
    
    if not instances:
        print("No instances loaded. Check JobShopLib installation.")
        return
    
    print(f"Loaded {len(instances)} instances")
    
    # Validate instances
    print("\n2. Validating instances...")
    validation_results = loader.validate_all_instances(instances)
    print("Validation results saved to data/processed/validation_results.csv")
    
    # Test classical solver
    print("\n3. Testing classical solver...")
    solver = ClassicalSolver()
    
    for instance in instances[:2]:  # Test first 2 instances
        print(f"\nSolving {instance.name} ({instance.num_jobs}x{instance.num_machines})...")
        
        params = SolverParams(
            solver_type=SolverType.DISPATCHING_RULE,
            seed=42
        )
        
        try:
            schedule, metrics = solver.solve_with_dispatching_rule(instance, params, rule="SPT")
            print(f"  Makespan: {metrics.makespan}")
            print(f"  Runtime: {metrics.runtime_seconds:.3f}s")
            if metrics.approximation_ratio:
                print(f"  Approx ratio: {metrics.approximation_ratio:.3f}")
        except Exception as e:
            print(f"  Solver failed: {e}")
    
    print("\n=== Prototype Complete ===")
    print("Next steps:")
    print("- Install dependencies: uv add job-shop-lib qiskit-optimization")
    print("- Run: python scripts/quick_start.py")
    print("- Check data/processed/ for exported instances")


if __name__ == "__main__":
    main()
