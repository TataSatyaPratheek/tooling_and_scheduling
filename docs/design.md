# Design — Native Starjob + Quantum Pipeline

## Streamlined Architecture
HuggingFace Dataset → Direct Processing → QAOA → Results
↓ ↓ ↓ ↓
Native format Size filtering Quantum Direct comparison
(no conversion) (.filter()) circuits (vs optimal_makespan)

## Native Data Structure
**Work directly with Starjob format:**
Native Starjob record structure:
{
"instance_id": int,
"job_count": int,
"machine_count": int,
"jobs": [
{
"processing_times": [int, ...],
"machine_sequence": [int, ...]
}, ...
],
"optimal_makespan": int
}


## Direct Processing Pipeline
- **Loading**: `datasets.load_dataset()` → work with dataset['train'] directly
- **Filtering**: `dataset.filter(lambda x: x['job_count'] <= 10)` for size-based subsets  
- **Analysis**: Compute statistics directly from dataset fields
- **Export**: Save raw records as JSON when inspection needed

## Quantum Circuit Construction
**Build QUBO directly from Starjob data:**
def build_qubo_from_starjob(record):
jobs = record['jobs']
for job_idx, job in enumerate(jobs):
for op_idx, (machine, duration) in enumerate(
zip(job['machine_sequence'], job['processing_times'])
):
# Create variables x[job_idx, op_idx, time] directly

## Performance Benchmarking
- **Baseline**: Compare directly against record['optimal_makespan']
- **Metrics**: Use native instance_id for result tracking
- **Scaling**: Filter by job_count/machine_count for size-based analysis
- **Reproducibility**: Log dataset commit hash and filtering parameters

## Experiment Tracking
- Log native Starjob metadata (instance_id, job_count, machine_count)
- Track quantum metrics (circuit depth, gate count) per instance
- Ensure identical runs via dataset versioning and filters.