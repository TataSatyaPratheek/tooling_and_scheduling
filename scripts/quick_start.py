#!/usr/bin/env python3
"""Starjob Dataset + QAOA Quick Start"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tooling_and_scheduling.loaders.starjob_loader import StarjobLoader  # Fixed naming!
from tooling_and_scheduling.solvers.classical_direct import ClassicalSolver
from tooling_and_scheduling.solvers.batch_processor import BatchProcessor


def main():
    print("=== T-1: Starjob Integration ===")
    
    # Load modern data
    loader = StarjobLoader()
    print("\n1. Analyzing Starjob dataset structure...")
    analysis = loader.analyze_dataset_structure()
    print(f"📊 Total instances: {analysis['total_instances']}")
    print(f"📊 Small (≤10x10): {analysis['size_distribution']['small_10x10']}")
    
    # Sample for rapid prototyping
    print("\n2. Sampling instances for prototyping...")
    instances = loader.sample_instances_by_size(samples_per_category=10)
    
    print(f"\n✅ Loaded {len(instances)} small instances for testing")
    print("🚀 Next: Classical baseline evaluation")
    
    print("\n=== T-2: Classical Baselines ===")
    
    # Test classical solver on single instance
    print("\n1. Testing classical solver on single instance...")
    sample_record = instances[0]
    sample_record = loader.convert_to_solver_format(sample_record)
    solver = ClassicalSolver()
    spt_result = solver.solve_shortest_processing_time(sample_record)
    metrics = solver.compare_to_optimal(spt_result, sample_record)
    print(f"✅ SPT solver test: approximation ratio {metrics['approximation_ratio']:.3f}")
    
    # Run batch processing on small sample
    print("\n2. Running batch processing on small sample...")
    processor = BatchProcessor()
    small_sample = instances[:10]  # First 10 instances
    # Convert all records to solver format
    small_sample = [loader.convert_to_solver_format(record) for record in small_sample]
    results_df = processor.run_classical_comparison(small_sample, methods=['SPT', 'LPT'])
    processor.export_results(results_df)
    
    # Show T-2 success metrics
    print(f"\n✅ Classical baselines tested on {len(small_sample)} instances")
    print(f"📊 Average approximation ratio: {results_df['approximation_ratio'].mean():.3f}")
    print(f"🎯 Optimal solutions found: {(results_df['approximation_ratio'] == 1.0).sum()}")
    print(f"💾 Results saved to: experiments/runs/classical_baseline_results.csv")
    
    print("\n🚀 Ready for T-3 (Direct QUBO Construction)")


if __name__ == "__main__":
    main()
