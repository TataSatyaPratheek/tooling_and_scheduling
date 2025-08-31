"""
Quantum Algorithm Metrics and Logging
Handles QAOA result persistence and approximation ratio calculations
"""
from typing import List, Dict, Any, Optional, Union
import os
import pandas as pd
import json
from pathlib import Path

__all__ = [
    'log_qaoa_results',
    'compute_approximation_ratio',
    'load_qaoa_results',
    'save_qaoa_results_json',
    'compute_statistics',
    'print_summary_stats',
    'log_single_qaoa_result',
    'batch_log_qaoa_results'
]


def log_qaoa_results(
    run_rows: List[Dict[str, Any]],
    out_csv: str = 'experiments/runs/qaoa_results.csv',
    append: bool = True
) -> None:
    """
    Log QAOA results to CSV file.

    Args:
        run_rows: List of dictionaries containing QAOA run results
        out_csv: Output CSV file path
        append: If True, append to existing file; if False, overwrite

    Creates file if missing, else appends. Schema includes:
    - instance_id, num_jobs, num_machines
    - reps, shots, depth, runtime, objective_value
    - seed, method, qubits
    - approximation_ratio (computed if optimum available)
    - timestamp
    """
    if not run_rows:
        return

    # Ensure output directory exists
    output_path = Path(out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for DataFrame
    processed_rows = []
    for row in run_rows:
        processed_row = {
            'instance_id': row.get('instance_id'),
            'num_jobs': row.get('num_jobs'),
            'num_machines': row.get('num_machines'),
            'optimal_makespan': row.get('optimal_makespan'),
            'reps': row.get('reps'),
            'shots': row.get('shots') or row.get('num_shots'),
            'circuit_depth': row.get('circuit_depth'),
            'run_time': row.get('run_time'),
            'transpile_time': row.get('transpile_time'),
            'objective_value': row.get('objective_value'),
            'seed': row.get('seed'),
            'method': row.get('method') or row.get('sampler_backend_method'),
            'num_qubits': row.get('num_qubits'),
            'solution_bitstring': row.get('solution_bitstring'),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Compute approximation ratio if optimum is available
        if (processed_row['objective_value'] is not None and
            processed_row['optimal_makespan'] is not None and
            processed_row['optimal_makespan'] > 0):
            processed_row['approximation_ratio'] = compute_approximation_ratio(
                processed_row['objective_value'],
                processed_row['optimal_makespan']
            )
        else:
            processed_row['approximation_ratio'] = None

        processed_rows.append(processed_row)

    # Create DataFrame
    df = pd.DataFrame(processed_rows)

    # Handle file operations
    if append and output_path.exists():
        try:
            # Read existing file to check schema compatibility
            existing_df = pd.read_csv(output_path)

            # Ensure all required columns exist in existing file
            required_cols = [
                'instance_id', 'num_jobs', 'num_machines', 'optimal_makespan',
                'reps', 'shots', 'circuit_depth', 'run_time', 'transpile_time',
                'objective_value', 'seed', 'method', 'num_qubits',
                'solution_bitstring', 'approximation_ratio', 'timestamp'
            ]

            for col in required_cols:
                if col not in existing_df.columns:
                    existing_df[col] = None

            # Append new data
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(output_path, index=False)

        except Exception as e:
            print(f"Warning: Could not append to existing file {output_path}: {e}")
            print("Creating new file instead...")
            df.to_csv(output_path, index=False)
    else:
        # Create new file or overwrite existing
        df.to_csv(output_path, index=False)

    print(f"Logged {len(run_rows)} QAOA results to {output_path}")


def compute_approximation_ratio(
    objective_value: Union[float, int],
    optimum: Union[float, int],
    minimize: bool = True
) -> Optional[float]:
    """
    Compute approximation ratio.

    By default the metric is for minimization problems (makespan):
      approximation_ratio = objective_value / optimum

    If minimize is False (maximization) the ratio is computed as:
      approximation_ratio = optimum / objective_value

    Args:
        objective_value: The achieved objective value
        optimum: The best known optimum value
        minimize: True if problem is minimization (default)

    Returns:
        Approximation ratio, or None if computation not possible
    """
    # Input validation
    if objective_value is None or optimum is None:
        return None

    # Handle NaN values
    try:
        if pd.isna(objective_value) or pd.isna(optimum):
            return None
    except (TypeError, ValueError):
        pass

    # Convert to float for safety
    try:
        obj_val = float(objective_value)
        opt_val = float(optimum)
    except (TypeError, ValueError):
        return None

    # Check for invalid optimum values
    if opt_val <= 0 or (not minimize and obj_val <= 0):
        return None

    # Compute ratio with safe handling
    try:
        if minimize:
            ratio = obj_val / opt_val
        else:
            ratio = opt_val / obj_val
        return float(ratio)
    except (ZeroDivisionError, OverflowError):
        return None


def load_qaoa_results(csv_path: str = 'experiments/runs/qaoa_results.csv') -> pd.DataFrame:
    """
    Load QAOA results from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with QAOA results, or empty DataFrame if file doesn't exist
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} does not exist")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)

        # Convert timestamp to datetime if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.DataFrame()


def save_qaoa_results_json(
    run_rows: List[Dict[str, Any]],
    out_json: str = 'experiments/runs/qaoa_results.json',
    append: bool = True
) -> None:
    """
    Save QAOA results to JSON file.

    Args:
        run_rows: List of dictionaries containing QAOA run results
        out_json: Output JSON file path
        append: If True, append to existing file; if False, overwrite
    """
    output_path = Path(out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp to each row
    timestamped_rows = []
    for row in run_rows:
        row_copy = row.copy()
        row_copy['timestamp'] = pd.Timestamp.now().isoformat()
        timestamped_rows.append(row_copy)

    if append and output_path.exists():
        try:
            # Load existing data
            with open(output_path, 'r') as f:
                existing_data = json.load(f)

            # Append new data
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
            existing_data.extend(timestamped_rows)

            # Save combined data
            with open(output_path, 'w') as f:
                json.dump(existing_data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not append to existing JSON file {output_path}: {e}")
            print("Creating new file instead...")
            with open(output_path, 'w') as f:
                json.dump(timestamped_rows, f, indent=2)
    else:
        # Create new file or overwrite existing
        with open(output_path, 'w') as f:
            json.dump(timestamped_rows, f, indent=2)

    print(f"Saved {len(run_rows)} QAOA results to {output_path}")


def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistics from QAOA results DataFrame.

    Args:
        df: DataFrame with QAOA results

    Returns:
        Dictionary with computed statistics
    """
    if df.empty:
        return {}

    stats = {}

    # Basic counts
    stats['total_runs'] = len(df)
    stats['unique_instances'] = int(df['instance_id'].nunique()) if 'instance_id' in df.columns else 0

    # Performance metrics
    if 'objective_value' in df.columns:
        valid_objectives = pd.to_numeric(df['objective_value'], errors='coerce').dropna()
        if not valid_objectives.empty:
            stats['objective_mean'] = float(valid_objectives.mean())
            stats['objective_std'] = float(valid_objectives.std())
            stats['objective_min'] = float(valid_objectives.min())
            stats['objective_max'] = float(valid_objectives.max())

    # Approximation ratio statistics
    if 'approximation_ratio' in df.columns:
        valid_ratios = pd.to_numeric(df['approximation_ratio'], errors='coerce').dropna()
        if not valid_ratios.empty:
            stats['approx_ratio_mean'] = float(valid_ratios.mean())
            stats['approx_ratio_std'] = float(valid_ratios.std())
            stats['approx_ratio_min'] = float(valid_ratios.min())
            stats['approx_ratio_max'] = float(valid_ratios.max())

    # Runtime statistics
    if 'run_time' in df.columns:
        valid_runtimes = pd.to_numeric(df['run_time'], errors='coerce').dropna()
        if not valid_runtimes.empty:
            stats['runtime_mean'] = float(valid_runtimes.mean())
            stats['runtime_std'] = float(valid_runtimes.std())

    # Group by parameters
    if 'reps' in df.columns:
        stats['reps_distribution'] = df['reps'].value_counts(dropna=False).to_dict()

    if 'shots' in df.columns:
        stats['shots_distribution'] = df['shots'].value_counts(dropna=False).to_dict()

    return stats


def print_summary_stats(df: pd.DataFrame) -> None:
    """
    Print summary statistics of QAOA results.

    Args:
        df: DataFrame with QAOA results
    """
    if df.empty:
        print("No data available")
        return

    stats = compute_statistics(df)

    print("=== QAOA Results Summary ===")
    print(f"Total runs: {stats.get('total_runs', 0)}")
    print(f"Unique instances: {stats.get('unique_instances', 0)}")

    if 'objective_mean' in stats:
        print("\nObjective Value:")
        print(f"  mean: {stats.get('objective_mean'):.3f}")
        print(f"  std : {stats.get('objective_std'):.3f}")
        print(f"  min : {stats.get('objective_min'):.3f}")
        print(f"  max : {stats.get('objective_max'):.3f}")

    if 'approx_ratio_mean' in stats:
        print("\nApproximation Ratio:")
        print(f"  mean: {stats.get('approx_ratio_mean'):.3f}")
        print(f"  std : {stats.get('approx_ratio_std'):.3f}")
        print(f"  min : {stats.get('approx_ratio_min'):.3f}")
        print(f"  max : {stats.get('approx_ratio_max'):.3f}")

    if 'runtime_mean' in stats:
        print("\nRuntime (s):")
        print(f"  mean: {stats.get('runtime_mean'):.3f}")
        print(f"  std : {stats.get('runtime_std'):.3f}")

    print("\nParameter distributions:")
    if 'reps_distribution' in stats:
        print(f"  Reps: {stats['reps_distribution']}")
    if 'shots_distribution' in stats:
        print(f"  Shots: {stats['shots_distribution']}")


# Convenience functions for integration with QAOA results
def log_single_qaoa_result(
    result: Any,
    out_csv: str = 'experiments/runs/qaoa_results.csv',
    append: bool = True
) -> None:
    """
    Log a single QAOA result.

    Args:
        result: QAOA result object (with attributes) or dict
        out_csv: Output CSV file path
        append: Whether to append to existing file
    """
    # Convert result to dict
    if hasattr(result, '__dict__'):
        # It's an object with attributes
        row = vars(result)
    elif isinstance(result, dict):
        row = result
    else:
        raise ValueError("Result must be a dict or object with attributes")

    log_qaoa_results([row], out_csv, append)


def batch_log_qaoa_results(
    results: List[Any],
    out_csv: str = 'experiments/runs/qaoa_results.csv',
    batch_size: int = 100
) -> None:
    """
    Log multiple QAOA results in batches to avoid memory issues.

    Args:
        results: List of QAOA result objects or dicts
        out_csv: Output CSV file path
        batch_size: Number of results to log at once
    """
    for i in range(0, len(results), batch_size):
        batch = results[i:i + batch_size]

        # Convert to dicts if needed
        batch_rows = []
        for result in batch:
            if hasattr(result, '__dict__'):
                batch_rows.append(vars(result))
            elif isinstance(result, dict):
                batch_rows.append(result)
            else:
                print(f"Warning: Skipping invalid result type: {type(result)}")
                continue

        log_qaoa_results(batch_rows, out_csv, append=True)
