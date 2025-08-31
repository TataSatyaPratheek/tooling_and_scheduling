import pytest
import pandas as pd
from unittest.mock import Mock
from tooling_and_scheduling.solvers.classical_direct import ClassicalSolver
from tooling_and_scheduling.solvers.batch_processor import BatchProcessor
from experiments.classical_analysis import ClassicalAnalyzer


@pytest.fixture
def mock_starjob_record():
    """Mock Starjob record with known optimal makespan"""
    return {
        "instance_id": 123,
        "job_count": 2,
        "machine_count": 2,
        "jobs": [
            {"processing_times": [3, 2], "machine_sequence": [0, 1]},
            {"processing_times": [2, 4], "machine_sequence": [1, 0]}
        ],
        "optimal_makespan": 7  # Known optimal for this instance
    }


@pytest.fixture
def mock_dataset_subset():
    """Mock dataset subset with 3 small instances"""
    return [
        {
            "instance_id": 1,
            "job_count": 2,
            "machine_count": 2,
            "jobs": [
                {"processing_times": [1, 2], "machine_sequence": [0, 1]},
                {"processing_times": [2, 1], "machine_sequence": [1, 0]}
            ],
            "optimal_makespan": 4
        },
        {
            "instance_id": 2,
            "job_count": 2,
            "machine_count": 2,
            "jobs": [
                {"processing_times": [3, 1], "machine_sequence": [0, 1]},
                {"processing_times": [1, 3], "machine_sequence": [1, 0]}
            ],
            "optimal_makespan": 5
        },
        {
            "instance_id": 3,
            "job_count": 2,
            "machine_count": 2,
            "jobs": [
                {"processing_times": [2, 3], "machine_sequence": [0, 1]},
                {"processing_times": [3, 2], "machine_sequence": [1, 0]}
            ],
            "optimal_makespan": 6
        }
    ]


@pytest.fixture
def mock_results_df():
    """Mock results DataFrame from batch processing"""
    data = {
        'instance_id': [1, 1, 2, 2, 3, 3],
        'method': ['SPT', 'LPT', 'SPT', 'LPT', 'SPT', 'LPT'],
        'makespan': [4, 5, 6, 7, 7, 8],
        'optimal_makespan': [4, 4, 5, 5, 6, 6],
        'approximation_ratio': [1.0, 1.25, 1.2, 1.4, 1.166, 1.333],
        'runtime_seconds': [0.01, 0.01, 0.02, 0.02, 0.03, 0.03],
        'job_count': [2, 2, 2, 2, 2, 2],
        'machine_count': [2, 2, 2, 2, 2, 2]
    }
    return pd.DataFrame(data)


def test_classical_solver_direct_format(mock_starjob_record):
    """Test that ClassicalSolver works directly on Starjob format"""
    solver = ClassicalSolver()
    
    # Test SPT
    spt_result = solver.solve_shortest_processing_time(mock_starjob_record)
    
    # Verify return structure
    assert 'makespan' in spt_result
    assert 'schedule' in spt_result
    assert 'runtime_seconds' in spt_result
    assert isinstance(spt_result['schedule'], list)
    assert len(spt_result['schedule']) > 0
    
    # Verify schedule format: (job_id, op_id, machine_id, start_time, end_time)
    for operation in spt_result['schedule']:
        assert len(operation) == 5
        job_id, op_id, machine_id, start_time, end_time = operation
        assert isinstance(job_id, int)
        assert isinstance(op_id, int)
        assert isinstance(machine_id, int)
        assert isinstance(start_time, (int, float))
        assert isinstance(end_time, (int, float))
        assert end_time >= start_time
    
    # Test LPT
    lpt_result = solver.solve_longest_processing_time(mock_starjob_record)
    assert 'makespan' in lpt_result
    assert 'schedule' in lpt_result
    
    # Test FCFS
    fcfs_result = solver.solve_first_come_first_served(mock_starjob_record)
    assert 'makespan' in fcfs_result
    assert 'schedule' in fcfs_result


def test_approximation_ratio_calculation(mock_starjob_record):
    """Test compare_to_optimal method with known results"""
    solver = ClassicalSolver()
    
    # Get SPT result
    spt_result = solver.solve_shortest_processing_time(mock_starjob_record)
    
    # Compare to optimal
    metrics = solver.compare_to_optimal(spt_result, mock_starjob_record)
    
    # Verify structure
    assert 'approximation_ratio' in metrics
    assert 'makespan_gap' in metrics
    assert 'is_optimal' in metrics
    
    # Verify calculation
    expected_ratio = spt_result['makespan'] / mock_starjob_record['optimal_makespan']
    assert abs(metrics['approximation_ratio'] - expected_ratio) < 1e-6
    
    # Test optimal case (ratio = 1.0)
    optimal_result = spt_result.copy()
    optimal_result['makespan'] = mock_starjob_record['optimal_makespan']
    optimal_metrics = solver.compare_to_optimal(optimal_result, mock_starjob_record)
    assert optimal_metrics['approximation_ratio'] == 1.0
    assert optimal_metrics['is_optimal'] == True
    
    # Test poor solution case
    poor_result = spt_result.copy()
    poor_result['makespan'] = mock_starjob_record['optimal_makespan'] * 2
    poor_metrics = solver.compare_to_optimal(poor_result, mock_starjob_record)
    assert poor_metrics['approximation_ratio'] == 2.0
    assert poor_metrics['is_optimal'] == False


def test_batch_processing_direct(mock_dataset_subset):
    """Test BatchProcessor works directly with Starjob records"""
    processor = BatchProcessor()
    
    # Run batch comparison
    results_df = processor.run_classical_comparison(mock_dataset_subset, methods=['SPT', 'LPT'])
    
    # Verify DataFrame structure
    required_cols = ['instance_id', 'method', 'makespan', 'optimal_makespan', 
                    'approximation_ratio', 'runtime_seconds', 'job_count', 'machine_count']
    for col in required_cols:
        assert col in results_df.columns
    
    # Verify data integrity
    assert len(results_df) == len(mock_dataset_subset) * 2  # 3 instances * 2 methods
    assert all(results_df['approximation_ratio'] > 0)
    assert all(results_df['runtime_seconds'] >= 0)
    
    # Verify instance_ids are preserved
    instance_ids = results_df['instance_id'].unique()
    expected_ids = [record['instance_id'] for record in mock_dataset_subset]
    assert set(instance_ids) == set(expected_ids)


def test_performance_analysis(mock_results_df):
    """Test ClassicalAnalyzer methods with mock results"""
    analyzer = ClassicalAnalyzer()
    
    # Test size analysis
    size_analysis = analyzer.analyze_performance_by_size(mock_results_df)
    assert 'small' in size_analysis
    assert 'SPT' in size_analysis['small']
    assert 'mean_ratio' in size_analysis['small']['SPT']
    assert 'std_ratio' in size_analysis['small']['SPT']
    
    # Test hard instance identification
    hard_instances = analyzer.identify_hard_instances(mock_results_df, threshold=1.3)
    assert isinstance(hard_instances, list)
    if hard_instances:
        for instance in hard_instances:
            assert 'instance_id' in instance
            assert 'why_difficult' in instance
    
    # Test optimal solution analysis
    optimality_stats = analyzer.optimal_solution_analysis(mock_results_df)
    assert 'optimal_count' in optimality_stats
    assert 'easy_instance_patterns' in optimality_stats
    assert 'hard_instance_patterns' in optimality_stats
    
    # Test report generation
    report = analyzer.generate_t2_completion_report()
    assert 't2_acceptance_criteria' in report
    assert 'summary' in report
    assert len(report['t2_acceptance_criteria']) > 0
