import heapq
import time
from typing import Dict, Any, List, Tuple


class ClassicalSolver:
    def solve_shortest_processing_time(self, starjob_record: Dict[str, Any]) -> Dict[str, Any]:
        return self._solve_with_dispatch(starjob_record, priority_func=lambda pt: pt)

    def solve_longest_processing_time(self, starjob_record: Dict[str, Any]) -> Dict[str, Any]:
        return self._solve_with_dispatch(starjob_record, priority_func=lambda pt: -pt)

    def solve_first_come_first_served(self, starjob_record: Dict[str, Any]) -> Dict[str, Any]:
        return self._solve_with_dispatch(starjob_record, priority_func=lambda pt, job_id, op_id: (job_id, op_id))

    def _solve_with_dispatch(self, starjob_record: Dict[str, Any], priority_func) -> Dict[str, Any]:
        start_time = time.time()
        
        jobs = starjob_record['jobs']
        machine_count = starjob_record['machine_count']
        num_jobs = len(jobs)
        
        # Initialize
        machine_available = [0] * machine_count
        job_available = [0] * num_jobs
        job_progress = [0] * num_jobs
        schedule: List[Tuple[int, int, int, int, int]] = []
        
        # Ready queue: (priority, job_id, op_id)
        ready_queue = []
        
        # Add initial operations
        for job_id in range(num_jobs):
            if jobs[job_id]['processing_times']:
                pt = jobs[job_id]['processing_times'][0]
                priority = priority_func(pt, job_id, 0) if callable(priority_func) and len(priority_func.__code__.co_varnames) > 1 else priority_func(pt)
                heapq.heappush(ready_queue, (priority, job_id, 0))
        
        while ready_queue:
            # Select next operation
            _, job_id, op_id = heapq.heappop(ready_queue)
            pt = jobs[job_id]['processing_times'][op_id]
            machine = jobs[job_id]['machine_sequence'][op_id]
            
            # Calculate start and end times
            start = max(job_available[job_id], machine_available[machine])
            end = start + pt
            
            # Record schedule
            schedule.append((job_id, op_id, machine, start, end))
            
            # Update availabilities
            job_available[job_id] = end
            machine_available[machine] = end
            
            # Add next operation if exists
            next_op = op_id + 1
            if next_op < len(jobs[job_id]['processing_times']):
                next_pt = jobs[job_id]['processing_times'][next_op]
                next_priority = priority_func(next_pt, job_id, next_op) if callable(priority_func) and len(priority_func.__code__.co_varnames) > 1 else priority_func(next_pt)
                heapq.heappush(ready_queue, (next_priority, job_id, next_op))
        
        makespan = max(machine_available) if machine_available else 0
        runtime_seconds = time.time() - start_time
        
        return {
            "makespan": makespan,
            "schedule": schedule,
            "runtime_seconds": runtime_seconds
        }

    def compare_to_optimal(self, schedule_result: Dict[str, Any], starjob_record: Dict[str, Any]) -> Dict[str, Any]:
        makespan = schedule_result["makespan"]
        optimal = starjob_record["optimal_makespan"]
        approximation_ratio = makespan / optimal if optimal > 0 else float('inf')
        makespan_gap = makespan - optimal
        is_optimal = makespan == optimal
        
        return {
            "approximation_ratio": approximation_ratio,
            "makespan_gap": makespan_gap,
            "is_optimal": is_optimal
        }