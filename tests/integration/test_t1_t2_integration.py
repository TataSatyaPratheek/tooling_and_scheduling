import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from tooling_and_scheduling.loaders.starjob_loader import StarjobLoader
from tooling_and_scheduling.solvers.classical_direct import ClassicalSolver
from tooling_and_scheduling.solvers.batch_processor import BatchProcessor


@pytest.fixture
def mock_starjob_loader():
    """Mock StarjobLoader for integration testing"""
    loader = MagicMock()
    
    # Mock analysis
    loader.analyze_dataset_structure.return_value = {
        'total_instances': 100,
        'size_distribution': {'small_10x10': 50}
    }
    
    # Mock sample instances
    mock_instances = [
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
        }
    ]
    loader.sample_instances_by_size.return_value = mock_instances
    
    return loader


def test_t1_t2_integration_components(mock_starjob_loader):
    """Integration test combining T-1 dataset loading with T-2 classical solvers"""
    # Get mock instances
    instances = mock_starjob_loader.sample_instances_by_size.return_value
    
    # Test T-1: Dataset loading
    assert len(instances) == 2
    assert all('instance_id' in instance for instance in instances)
    assert all('jobs' in instance for instance in instances)
    
    # Test T-2: Classical solvers
    solver = ClassicalSolver()
    processor = BatchProcessor()
    
    # Test single solver
    sample_record = instances[0]
    spt_result = solver.solve_shortest_processing_time(sample_record)
    assert spt_result['makespan'] > 0
    assert len(spt_result['schedule']) > 0
    
    metrics = solver.compare_to_optimal(spt_result, sample_record)
    assert 'approximation_ratio' in metrics
    assert metrics['approximation_ratio'] >= 1.0
    
    # Test batch processing
    results_df = processor.run_classical_comparison(instances, methods=['SPT', 'LPT', 'FCFS'])
    assert len(results_df) == len(instances) * 3  # 2 instances * 3 methods
    assert all(results_df['approximation_ratio'] >= 1.0)
    assert all(results_df['runtime_seconds'] >= 0)
    
    # Verify instance IDs preserved
    instance_ids = results_df['instance_id'].unique()
    expected_ids = [record['instance_id'] for record in instances]
    assert set(instance_ids) == set(expected_ids)


def test_end_to_end_classical_pipeline(mock_starjob_loader):
    """End-to-end test of classical solver pipeline"""
    # Get mock instances
    instances = mock_starjob_loader.sample_instances_by_size.return_value
    
    # Test single solver
    solver = ClassicalSolver()
    sample_record = instances[0]
    
    spt_result = solver.solve_shortest_processing_time(sample_record)
    assert spt_result['makespan'] > 0
    assert len(spt_result['schedule']) > 0
    
    metrics = solver.compare_to_optimal(spt_result, sample_record)
    assert 'approximation_ratio' in metrics
    assert metrics['approximation_ratio'] >= 1.0
    
    # Test batch processing
    processor = BatchProcessor()
    results_df = processor.run_classical_comparison(instances, methods=['SPT', 'LPT', 'FCFS'])
    
    # Verify results
    assert len(results_df) == len(instances) * 3  # 2 instances * 3 methods
    assert all(results_df['approximation_ratio'] >= 1.0)
    assert all(results_df['runtime_seconds'] >= 0)
    
    # Verify all methods are tested
    methods_used = results_df['method'].unique()
    assert set(methods_used) == {'SPT', 'LPT', 'FCFS'}
    
    # Verify instance IDs preserved
    instance_ids = results_df['instance_id'].unique()
    expected_ids = [record['instance_id'] for record in instances]
    assert set(instance_ids) == set(expected_ids)


def test_classical_solver_consistency():
    """Test that classical solvers produce consistent results for the same input"""
    # Create a simple test instance
    test_record = {
        "instance_id": 999,
        "job_count": 2,
        "machine_count": 2,
        "jobs": [
            {"processing_times": [2, 3], "machine_sequence": [0, 1]},
            {"processing_times": [1, 4], "machine_sequence": [1, 0]}
        ],
        "optimal_makespan": 6
    }
    
    solver = ClassicalSolver()
    
    # Run multiple times to check consistency
    results = []
    for _ in range(3):
        spt_result = solver.solve_shortest_processing_time(test_record)
        results.append(spt_result['makespan'])
    
    # All results should be identical (deterministic)
    assert all(r == results[0] for r in results)
    
    # Test all methods produce valid results
    methods = ['SPT', 'LPT', 'FCFS']
    for method in methods:
        if method == 'SPT':
            result = solver.solve_shortest_processing_time(test_record)
        elif method == 'LPT':
            result = solver.solve_longest_processing_time(test_record)
        elif method == 'FCFS':
            result = solver.solve_first_come_first_served(test_record)
        
        assert result['makespan'] > 0
        assert len(result['schedule']) == 4  # 2 jobs * 2 operations each
        assert result['runtime_seconds'] >= 0
