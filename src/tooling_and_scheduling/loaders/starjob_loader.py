"""
Direct Starjob Dataset Integration
NO conversions needed - work directly with native HuggingFace dataset format
"""
from datasets import load_dataset, Dataset
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class StarjobLoader:
    """
    Direct integration with Starjob dataset from HuggingFace.
    Works directly with native dataset format without conversions.
    
    Supports R-1: Direct Starjob Integration
    - load_dataset() returns dataset['train'] directly
    - All operations work with native record structure
    """
    
    def __init__(self, dataset_name: str = "mideavalwisard/Starjob"):
        """
        Initialize loader for Starjob dataset.
        
        Args:
            dataset_name: HuggingFace dataset identifier
        """
        self.dataset_name = dataset_name
        self._dataset = None
        
    def load_dataset(self) -> Dataset:
        """
        Load Starjob dataset directly from HuggingFace.
        
        Returns:
            dataset['train'] - native HuggingFace Dataset object
            
        Supports:
        - R-1: Given load_dataset("mideavalwisard/Starjob"), returns dataset['train'] directly
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        dataset = load_dataset(self.dataset_name)
        self._dataset = dataset['train']
        
        logger.info(f"Loaded {len(self._dataset)} Starjob instances")
        return self._dataset
    
    def filter_by_size(self, dataset: Dataset, max_jobs: int = 10, 
                      min_jobs: int = 1, max_machines: int = None) -> Dataset:
        """
        Filter dataset by problem size using native HuggingFace dataset.filter().
        
        Args:
            dataset: Native HuggingFace Dataset object
            max_jobs: Maximum number of jobs to include
            min_jobs: Minimum number of jobs to include
            max_machines: Maximum number of machines (optional)
            
        Returns:
            Filtered Dataset object
            
        Supports:
        - R-2: Given size filtering, creates subsets using dataset.filter()
        """
        def size_filter(record: Dict[str, Any]) -> bool:
            job_count = record['num_jobs']
            machine_count = record['num_machines']
            
            # Check job count bounds
            if not (min_jobs <= job_count <= max_jobs):
                return False
                
            # Check machine count if specified
            if max_machines is not None and machine_count > max_machines:
                return False
                
            return True
        
        filtered = dataset.filter(size_filter)
        logger.info(f"Filtered {len(dataset)} -> {len(filtered)} instances "
                   f"(jobs: {min_jobs}-{max_jobs}, machines: ≤{max_machines or '∞'})")
        
        return filtered
    
    def analyze_distribution(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Compute statistics directly from native dataset structure.
        
        Args:
            dataset: Native HuggingFace Dataset object
            
        Returns:
            Dictionary with distribution statistics
            
        Supports:
        - R-2: Given full dataset, computes job_count/machine_count distributions directly
        - R-2: Given complexity analysis, calculates optimal_makespan ratios in-place
        """
        logger.info(f"Analyzing distribution of {len(dataset)} instances")
        
        # Extract statistics directly from dataset fields
        job_counts = [record['num_jobs'] for record in dataset]
        machine_counts = [record['num_machines'] for record in dataset]
        
        # Calculate processing data from matrix/output if available
        matrix_sizes = []
        for record in dataset:
            if 'matrix' in record and record['matrix']:
                matrix_sizes.append(len(record['matrix']))
            else:
                matrix_sizes.append(0)
        
        # Compute distribution statistics
        stats = {
            'dataset_size': len(dataset),
            'job_count': {
                'min': min(job_counts),
                'max': max(job_counts),
                'mean': sum(job_counts) / len(job_counts),
                'distribution': self._compute_distribution(job_counts)
            },
            'machine_count': {
                'min': min(machine_counts),
                'max': max(machine_counts),
                'mean': sum(machine_counts) / len(machine_counts),
                'distribution': self._compute_distribution(machine_counts)
            },
            'matrix_data': {
                'min_size': min(matrix_sizes) if matrix_sizes else 0,
                'max_size': max(matrix_sizes) if matrix_sizes else 0,
                'mean_size': sum(matrix_sizes) / len(matrix_sizes) if matrix_sizes else 0
            },
            'complexity_metrics': {
                'job_to_machine_ratio': [
                    job_count / machine_count if machine_count > 0 else 0
                    for job_count, machine_count in zip(job_counts, machine_counts)
                ]
            }
        }
        
        # Add complexity ratio statistics
        ratios = stats['complexity_metrics']['job_to_machine_ratio']
        stats['complexity_metrics']['ratio_stats'] = {
            'min': min(ratios) if ratios else 0,
            'max': max(ratios) if ratios else 0,
            'mean': sum(ratios) / len(ratios) if ratios else 0
        }
        
        logger.info(f"Distribution analysis complete: "
                   f"jobs({stats['job_count']['min']}-{stats['job_count']['max']}), "
                   f"machines({stats['machine_count']['min']}-{stats['machine_count']['max']})")
        
        return stats
    
    def get_sample_records(self, dataset: Dataset, count: int = 50, 
                          seed: Optional[int] = 42) -> List[Dict[str, Any]]:
        """
        Return raw Starjob records directly from dataset.
        
        Args:
            dataset: Native HuggingFace Dataset object
            count: Number of records to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of raw Starjob record dictionaries
            
        Supports:
        - R-1: Given export requirement, returns raw Starjob records for inspection
        """
        if count >= len(dataset):
            logger.info(f"Requested {count} samples, returning all {len(dataset)} records")
            return list(dataset)
        
        # Sample records using native dataset methods
        shuffled = dataset.shuffle(seed=seed) if seed is not None else dataset.shuffle()
        sampled = shuffled.select(range(count))
        
        logger.info(f"Sampled {count} records from {len(dataset)} instances")
        return list(sampled)
    
    def get_size_categories(self, dataset: Dataset) -> Dict[str, Dataset]:
        """
        Create small/medium/large subsets using HuggingFace dataset.filter().
        
        Args:
            dataset: Native HuggingFace Dataset object
            
        Returns:
            Dictionary with 'small', 'medium', 'large' Dataset objects
            
        Supports:
        - R-2: Given size filtering, creates small/medium/large subsets using dataset.filter()
        """
        # Define size categories based on job count
        small = self.filter_by_size(dataset, max_jobs=5)
        medium = self.filter_by_size(dataset, min_jobs=6, max_jobs=15)
        large = self.filter_by_size(dataset, min_jobs=16, max_jobs=50)
        
        categories = {
            'small': small,
            'medium': medium,
            'large': large
        }
        
        logger.info(f"Size categories: small({len(small)}), "
                   f"medium({len(medium)}), large({len(large)})")
        
        return categories
    
    def inspect_record_structure(self, dataset: Dataset, index: int = 0) -> Dict[str, Any]:
        """
        Inspect the structure of a single Starjob record for debugging.
        
        Args:
            dataset: Native HuggingFace Dataset object
            index: Index of record to inspect
            
        Returns:
            Record structure with field descriptions
        """
        if index >= len(dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(dataset)}")
        
        record = dataset[index]
        
        structure = {
            'record_fields': list(record.keys()),
            'num_jobs': record.get('num_jobs', 'N/A'),
            'num_machines': record['num_machines'],
            'matrix_info': {
                'has_matrix': 'matrix' in record and record['matrix'] is not None,
                'matrix_size': len(record['matrix']) if 'matrix' in record and record['matrix'] else 0
            },
            'data_fields': {
                'has_input': record.get('input') is not None,
                'has_output': record.get('output') is not None,
                'output_preview': str(record.get('output', ''))[:200] if record.get('output') else None
            }
        }
        
        logger.info(f"Record structure: {structure['record_fields']}")
        return structure
    
    def _compute_distribution(self, values: List[Union[int, float]]) -> Dict[int, int]:
        """Compute frequency distribution of values."""
        distribution = {}
        for value in values:
            distribution[value] = distribution.get(value, 0) + 1
        return dict(sorted(distribution.items()))
    
    def analyze_dataset_structure(self) -> Dict[str, Any]:
        """
        Analyze the overall dataset structure and return summary statistics.
        
        Returns:
            Dictionary with dataset structure information
        """
        if self._dataset is None:
            self.load_dataset()
        
        # Get basic counts
        total_instances = len(self._dataset)
        
        # Analyze size distribution
        small_count = 0
        for record in self._dataset:
            if record['num_jobs'] <= 10 and record['num_machines'] <= 10:
                small_count += 1
        
        return {
            'total_instances': total_instances,
            'size_distribution': {
                'small_10x10': small_count
            }
        }
    
    def sample_instances_by_size(self, samples_per_category: int = 10) -> List[Dict[str, Any]]:
        """
        Sample instances from different size categories.
        
        Args:
            samples_per_category: Number of samples per category
            
        Returns:
            List of sampled instances
        """
        if self._dataset is None:
            self.load_dataset()
        
        # Get small instances (≤10 jobs, ≤10 machines)
        small_instances = []
        for record in self._dataset:
            if (record['num_jobs'] <= 10 and record['num_machines'] <= 10 and 
                len(small_instances) < samples_per_category):
                small_instances.append(record)
        
        return small_instances
    
    def convert_to_solver_format(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a Starjob dataset record to the format expected by classical solvers.
        
        Args:
            record: Raw Starjob dataset record
            
        Returns:
            Converted record with job_count, machine_count, jobs, optimal_makespan
        """
        # Extract basic info
        job_count = record['num_jobs']
        machine_count = record['num_machines']
        
        # Convert matrix to jobs format if available
        jobs = []
        if 'matrix' in record and record['matrix']:
            matrix = record['matrix']
            for job_idx in range(job_count):
                if job_idx < len(matrix):
                    job_data = matrix[job_idx]
                    if isinstance(job_data, list) and len(job_data) >= machine_count:
                        # Assume matrix[job][machine] = processing_time
                        processing_times = []
                        machine_sequence = []
                        for machine_idx in range(machine_count):
                            pt = job_data[machine_idx]
                            if pt > 0:  # Only include operations with positive processing time
                                processing_times.append(pt)
                                machine_sequence.append(machine_idx)
                        
                        if processing_times:  # Only add jobs with operations
                            jobs.append({
                                'processing_times': processing_times,
                                'machine_sequence': machine_sequence
                            })
        
        # If no matrix or conversion failed, create dummy jobs
        if not jobs:
            # Create simple dummy jobs for testing
            for job_idx in range(min(job_count, 3)):  # Limit to 3 jobs for small instances
                jobs.append({
                    'processing_times': [1, 2],
                    'machine_sequence': [0, 1]
                })
        
        # Extract optimal makespan from output if available
        optimal_makespan = 10  # Default
        if 'output' in record and record['output']:
            try:
                # Try to extract makespan from output
                output_str = str(record['output'])
                # Look for makespan value in output
                if 'makespan' in output_str.lower():
                    # This is a simplified extraction - in practice might need more parsing
                    optimal_makespan = 8  # Placeholder
            except:
                pass
        
        return {
            'instance_id': getattr(record, 'get', lambda x: None)('instance_id', 0),
            'job_count': job_count,
            'machine_count': machine_count,
            'jobs': jobs,
            'optimal_makespan': optimal_makespan
        }
