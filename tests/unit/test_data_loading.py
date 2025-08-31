# tests/unit/test_data_loading.py
import pytest
from pathlib import Path
from src.tooling_and_scheduling.data_loader import DataLoader
from src.tooling_and_scheduling.parsers.job_shop_parser import JobShopParser


def test_parser_initialization():
    """Test parser can be created and lists instances"""
    parser = JobShopParser()
    assert len(parser.available_instances) > 0
    assert "ft06" in parser.available_instances


def test_load_small_instance():
    """Test loading a known small instance"""
    parser = JobShopParser()
    instance = parser.load_instance("ft06")
    
    assert instance.name == "ft06"
    assert instance.num_jobs == 6
    assert instance.num_machines == 6
    assert len(instance.jobs) == 6
    assert len(instance.operations) == 36  # 6 jobs * 6 operations each


def test_data_loader():
    """Test data loader functionality"""
    data_dir = Path("test_data")
    loader = DataLoader(data_dir)
    
    # This should not fail even if instances aren't available
    instances = loader.load_small_instances()
    assert isinstance(instances, list)
    
    # Cleanup
    import shutil
    if data_dir.exists():
        shutil.rmtree(data_dir)


if __name__ == "__main__":
    # Quick test run
    test_parser_initialization()
    test_load_small_instance()
    print("All tests passed!")
