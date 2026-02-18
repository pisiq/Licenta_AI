"""
Quick test of data loading.
"""
from iclr_data_loader import ICLRDataLoader, run_sanity_checks

loader = ICLRDataLoader(base_path="C:/Facultate/Licenta/data")
train, dev, test = loader.load_all_splits(verbose=True)
run_sanity_checks(train, dev, test)

