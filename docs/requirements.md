# Requirements — Direct Starjob Dataset + QAOA Pipeline

**Data Source**: Native Starjob format from HuggingFace - work directly with dataset structure, no conversions needed.

## R-1 Direct Starjob Integration
When Starjob dataset is loaded, the system shall work directly with the native HuggingFace dataset format without unnecessary conversions.
Acceptance:
- Given `load_dataset("mideavalwisard/Starjob")`, when loaded, then system works directly with dataset['train'] records
- Given dataset analysis, when executed, then size distribution and complexity metrics are computed directly from native format
- Given export requirement, when needed, then raw Starjob records are saved as JSON for inspection

## R-2 Native Data Analysis  
When analyzing Starjob instances, the system shall compute statistics directly from the dataset structure without format conversion overhead.
Acceptance:
- Given full dataset, when analyzed, then job_count/machine_count distributions are computed directly from dataset fields
- Given complexity analysis, when executed, then optimal_makespan ratios and problem difficulty metrics are calculated in-place
- Given size filtering, when applied, then small/medium/large subsets are created using HuggingFace dataset.filter()

## R-3 Direct Classical Baselines
When classical solvers run, the system shall operate directly on Starjob format data structures.
Acceptance:
- Given Starjob instance record, when passed to solver, then processing_times and machine_sequence are used directly
- Given batch processing, when executed, then results include original instance_id and optimal_makespan comparisons

## R-4 Native QUBO Construction
When building QUBO formulations, the system shall construct variables directly from Starjob job/machine/time data without intermediate conversions.
Acceptance:
- Given Starjob record with jobs[{processing_times, machine_sequence}], when QUBO built, then variables x[job,op,time] are created directly
- Given time horizon calculation, when computed, then uses sum of processing_times + optimal_makespan bounds from dataset

## R-5 Streamlined Experiment Tracking
When experiments run, the system shall log native Starjob metadata alongside quantum metrics.
Acceptance:
- Given experiment run, when logged, then includes original instance_id, job_count, machine_count, and optimal_makespan from dataset
- Given reproducibility, when achieved, then HuggingFace dataset commit and filtering parameters ensure identical runs
