# Tasks — Streamlined Starjob Pipeline

## T-1 Direct Starjob Integration
- Implement native HuggingFace dataset loader (no conversion classes needed)
- Create size-based filtering using dataset.filter() methods  
- Add direct analysis functions working on raw dataset records
- Export raw Starjob samples as JSON for inspection (no CSV conversion)

## T-2 Native Classical Baselines  
- Implement dispatching rules working directly on Starjob record format
- Process jobs[i]['processing_times'] and jobs[i]['machine_sequence'] directly
- Compare results against record['optimal_makespan'] without conversion overhead

## T-3 Direct QUBO Construction
- Build QuadraticProgram variables directly from Starjob jobs structure
- Calculate time horizon from max(processing_times) + optimal_makespan bounds
- Create penalties using native job/machine indices from dataset

## T-4 Streamlined Experiments
- Log experiments with native instance_id, job_count, machine_count metadata
- Track HuggingFace dataset version and filtering parameters for reproducibility
- Generate results comparing quantum vs optimal_makespan directly

## T-5 Clean API Development
- Design FastAPI endpoints accepting/returning native Starjob format
- Build Streamlit UI displaying jobs/machines/times directly from dataset records
- No conversion layers - work with Starjob structure throughout
