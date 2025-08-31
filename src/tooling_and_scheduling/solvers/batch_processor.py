import pandas as pd
from .classical_direct import ClassicalSolver
from typing import Dict, Any, List


class BatchProcessor:
    def __init__(self):
        self.solver = ClassicalSolver()

    def run_classical_comparison(self, dataset_subset, methods: List[str] = ['SPT', 'LPT', 'FCFS']) -> pd.DataFrame:
        results = []
        
        for record in dataset_subset:
            instance_id = record['instance_id']
            job_count = record['job_count']
            machine_count = record['machine_count']
            optimal_makespan = record['optimal_makespan']
            
            for method in methods:
                if method == 'SPT':
                    result = self.solver.solve_shortest_processing_time(record)
                elif method == 'LPT':
                    result = self.solver.solve_longest_processing_time(record)
                elif method == 'FCFS':
                    result = self.solver.solve_first_come_first_served(record)
                else:
                    continue
                
                makespan = result['makespan']
                runtime_seconds = result['runtime_seconds']
                approximation_ratio = makespan / optimal_makespan if optimal_makespan > 0 else float('inf')
                
                results.append({
                    'instance_id': instance_id,
                    'method': method,
                    'makespan': makespan,
                    'optimal_makespan': optimal_makespan,
                    'approximation_ratio': approximation_ratio,
                    'runtime_seconds': runtime_seconds,
                    'job_count': job_count,
                    'machine_count': machine_count
                })
        
        df = pd.DataFrame(results)
        return df

    def analyze_performance_by_size(self, results_df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
        def categorize_size(job_count: int) -> str:
            if job_count <= 10:
                return 'small'
            elif job_count <= 20:
                return 'medium'
            else:
                return 'large'
        
        results_df = results_df.copy()
        results_df['size_category'] = results_df['job_count'].apply(categorize_size)
        
        size_analysis = {}
        for size in ['small', 'medium', 'large']:
            size_df = results_df[results_df['size_category'] == size]
            size_analysis[size] = {}
            for method in results_df['method'].unique():
                method_df = size_df[size_df['method'] == method]
                if not method_df.empty:
                    mean_ratio = method_df['approximation_ratio'].mean()
                    std_ratio = method_df['approximation_ratio'].std()
                    size_analysis[size][method] = {'mean_ratio': mean_ratio, 'std_ratio': std_ratio}
        
        return size_analysis

    def export_results(self, results_df: pd.DataFrame, output_path: str = "experiments/runs/classical_baseline_results.csv"):
        results_df.to_csv(output_path, index=False)

    def generate_summary_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        summary = {}
        
        # Best/worst approximation ratios per method
        summary['best_worst_ratios'] = {}
        for method in results_df['method'].unique():
            method_df = results_df[results_df['method'] == method]
            best = method_df['approximation_ratio'].min()
            worst = method_df['approximation_ratio'].max()
            summary['best_worst_ratios'][method] = {'best': best, 'worst': worst}
        
        # Runtime performance comparison
        runtime_stats = results_df.groupby('method')['runtime_seconds'].agg(['mean', 'std', 'min', 'max'])
        summary['runtime_performance'] = runtime_stats.to_dict()
        
        # Instances where classical methods achieved optimal solutions
        optimal_instances = {}
        for method in results_df['method'].unique():
            method_df = results_df[results_df['method'] == method]
            optimal_count = (method_df['makespan'] == method_df['optimal_makespan']).sum()
            total = len(method_df)
            percentage = optimal_count / total * 100 if total > 0 else 0
            optimal_instances[method] = {
                'optimal_count': optimal_count,
                'total': total,
                'percentage': percentage
            }
        summary['optimal_solutions'] = optimal_instances
        
        return summary