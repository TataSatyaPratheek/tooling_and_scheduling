# src/tooling_and_scheduling/data_loader.py
from pathlib import Path
import pandas as pd
from typing import List, Dict
from .parsers.job_shop_parser import JobShopParser
from .models.job_shop import JobShopInstance


class DataLoader:
    """Main data loading interface"""
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.parser = JobShopParser()
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_small_instances(self) -> List[JobShopInstance]:
        """Load 6x6 and small instances for rapid prototyping"""
        small_instances = ["ft06", "la01", "la02", "abz5"]  # Known small instances
        instances = []
        
        for name in small_instances:
            try:
                instance = self.parser.load_instance(name)
                if instance.num_jobs <= 10 and instance.num_machines <= 10:
                    instances.append(instance)
                    print(f"Loaded {name}: {instance.num_jobs}x{instance.num_machines}")
                    
                    # Export for inspection
                    self.parser.export_to_json(
                        instance, 
                        self.processed_dir / f"{name}.json"
                    )
                    self.parser.export_to_csv(
                        instance, 
                        self.processed_dir / name
                    )
            except Exception as e:
                print(f"Failed to load {name}: {e}")
        
        return instances
    
    def validate_all_instances(self, instances: List[JobShopInstance]) -> pd.DataFrame:
        """Run validation checks on all instances"""
        results = []
        
        for instance in instances:
            checks = self.parser.validate_instance(instance)
            result = {"instance": instance.name, **checks}
            results.append(result)
        
        validation_df = pd.DataFrame(results)
        validation_df.to_csv(self.processed_dir / "validation_results.csv", index=False)
        return validation_df
