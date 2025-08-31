#!/usr/bin/env python3
"""Starjob Dataset + QAOA Quick Start"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tooling_and_scheduling.loaders.starjob_loader import StarjobLoader  # Fixed naming!
from tooling_and_scheduling.solvers.classical_direct import ClassicalSolver
from tooling_and_scheduling.solvers.batch_processor import BatchProcessor

# Best-effort imports for quantum utilities (support multiple module locations)
try:
    from tooling_and_scheduling.quantum.qubo_builder import build_qubo_from_starjob
except Exception:
    try:
        from tooling_and_scheduling.solvers.quantum_qubo import build_qubo_from_starjob
    except Exception:
        build_qubo_from_starjob = None

try:
    from tooling_and_scheduling.quantum.qaoa_runner import run_qaoa_on_qubo
except Exception:
    try:
        from tooling_and_scheduling.solvers.quantum_qaoa import run_qaoa_on_qubo
    except Exception:
        run_qaoa_on_qubo = None

try:
    from tooling_and_scheduling.quantum.metrics import log_qaoa_results
except Exception:
    # Fallback to direct import path if package layout differs
    try:
        from tooling_and_scheduling.quantum.metrics import log_qaoa_results
    except Exception:
        log_qaoa_results = None


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

    # === T-3: Direct QUBO Construction & QAOA Demo ===
    print("\n=== T-3: Direct QUBO Construction & QAOA Demo ===")

    # Pick a tiny original instance (not converted solver format) to build a compact QUBO
    original_instance = instances[0] if instances else None

    if build_qubo_from_starjob is None or run_qaoa_on_qubo is None:
        print("Quantum utilities not found in the current environment. Skipping T-3.")
        return

    if original_instance is None:
        print("No instances available for T-3.")
        return

    try:
        # Convert to solver format before building QUBO (builder expects jobs/processing_times)
        print("Converting instance to solver format for QUBO builder...")
        converted_instance = loader.convert_to_solver_format(original_instance)

        # Try to build a compact QUBO (small horizon) for fast demo
        print("Building QUBO (horizon_method='auto', max_T=60) ...")
        qubo_result = build_qubo_from_starjob(converted_instance, horizon_method='auto', max_T=60)
        qubo = getattr(qubo_result, 'qubo', qubo_result)

        print("QUBO built. Running QAOA (reps=1,2; shots=1024)...")

        qaoa_rows = []
        for reps in [1, 2]:
            try:
                qres = run_qaoa_on_qubo(qubo, reps=reps, shots=1024, seed=42, method='automatic', starjob_metadata=converted_instance)
                # Convert result to dict for logging
                row = vars(qres) if hasattr(qres, '__dict__') else dict(qres)
                qaoa_rows.append(row)
                print(f"  reps={reps} objective={row.get('objective_value')} depth={row.get('circuit_depth')} runtime={row.get('run_time')}")
            except MemoryError:
                print(f"MemoryError during QAOA for reps={reps}. Try reducing 'max_T' or using a smaller instance.")
            except Exception as e:
                print(f"QAOA run failed for reps={reps}: {e}")

        # Log results if metrics available
        if log_qaoa_results is not None and qaoa_rows:
            try:
                log_qaoa_results(qaoa_rows)
                print(f"💾 QAOA results saved to experiments/runs/qaoa_results.csv")
            except Exception as e:
                print(f"Warning: could not save QAOA results: {e}")

        print("\n✅ T-3 complete: QUBO constructed and QAOA demo ran")

    except MemoryError:
        print("MemoryError: QUBO horizon too large. Try smaller max_T (e.g., 30) or reduce problem size.")
    except TypeError as te:
        # Some builder implementations may expect converted solver format
        print(f"TypeError building QUBO: {te}. Trying with converted solver format...")
        try:
            converted = loader.convert_to_solver_format(original_instance)
            qubo_result = build_qubo_from_starjob(converted, horizon_method='auto', max_T=60)
            qubo = getattr(qubo_result, 'qubo', qubo_result)
            # Rerun QAOA once with reps=1
            qres = run_qaoa_on_qubo(qubo, reps=1, shots=1024, seed=42, method='automatic', starjob_metadata=converted)
            row = vars(qres) if hasattr(qres, '__dict__') else dict(qres)
            if log_qaoa_results is not None:
                log_qaoa_results([row])
            print("✅ T-3 retried successfully with converted record")
        except Exception as e:
            print(f"Retry failed: {e}")
    except Exception as e:
        print(f"Failed to build QUBO or run QAOA: {e}")
        print("Hint: If the problem is too large, reduce 'max_T' or sample a smaller instance.")


if __name__ == "__main__":
    main()
