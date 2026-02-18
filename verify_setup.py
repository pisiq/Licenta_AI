"""
Setup verification script.

Run this after installing dependencies to verify everything is set up correctly.
"""
import sys


def check_imports():
    """Check if all required packages can be imported."""
    print("Checking package installations...")
    print("-" * 80)

    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sklearn': 'scikit-learn',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'scipy': 'SciPy',
        'tqdm': 'tqdm',
        'tensorboard': 'TensorBoard'
    }

    all_ok = True

    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20s} - OK")
        except ImportError:
            print(f"✗ {name:20s} - NOT FOUND")
            all_ok = False

    print("-" * 80)
    return all_ok


def check_project_files():
    """Check if all project files exist."""
    import os

    print("\nChecking project files...")
    print("-" * 80)

    required_files = [
        'config.py',
        'data_preprocessing.py',
        'model.py',
        'metrics.py',
        'trainer.py',
        'train.py',
        'inference.py',
        'example.py',
        'test_pipeline.py',
        'generate_sample_data.py',
        'requirements.txt',
        'README.md'
    ]

    all_ok = True

    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file:30s} - Found")
        else:
            print(f"✗ {file:30s} - Missing")
            all_ok = False

    print("-" * 80)
    return all_ok


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        print("\nChecking CUDA...")
        print("-" * 80)

        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
            print("  Consider installing CUDA for faster training")

        print("-" * 80)

    except ImportError:
        print("\n⚠ Cannot check CUDA - PyTorch not installed")


def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    print("-" * 80)

    try:
        # Test imports
        from config import ModelConfig, TrainingConfig, DataConfig
        from data_preprocessing import TextPreprocessor, ReviewAggregator
        from model import MultiTaskOrdinalClassifier
        from metrics import quadratic_weighted_kappa

        print("✓ Configuration classes - OK")
        print("✓ Data preprocessing - OK")
        print("✓ Model classes - OK")
        print("✓ Metrics - OK")

        # Test basic operations
        config = ModelConfig()
        print(f"✓ Model config created with {len(config.score_dimensions)} dimensions")

        preprocessor = TextPreprocessor()
        test_text = "This is a test paper with some   extra  spaces."
        cleaned = preprocessor.clean_text(test_text)
        print(f"✓ Text preprocessing works")

        aggregator = ReviewAggregator()
        reviews = [{"IMPACT": 3}, {"IMPACT": 4}]
        agg = aggregator.aggregate_scores(reviews)
        print(f"✓ Review aggregation works (result: {agg['IMPACT']})")

        print("-" * 80)
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        print("-" * 80)
        return False


def main():
    """Run all checks."""
    print("\n" + "=" * 80)
    print("SETUP VERIFICATION")
    print("=" * 80 + "\n")

    # Check packages
    packages_ok = check_imports()

    # Check files
    files_ok = check_project_files()

    # Check CUDA
    check_cuda()

    # Test functionality
    if packages_ok and files_ok:
        functionality_ok = test_basic_functionality()
    else:
        functionality_ok = False

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if packages_ok:
        print("✓ All required packages installed")
    else:
        print("✗ Some packages missing - run: pip install -r requirements.txt")

    if files_ok:
        print("✓ All project files present")
    else:
        print("✗ Some project files missing")

    if functionality_ok:
        print("✓ Basic functionality working")
    else:
        print("✗ Functionality tests failed")

    print("=" * 80 + "\n")

    if packages_ok and files_ok and functionality_ok:
        print("🎉 Setup verified successfully!")
        print("\nNext steps:")
        print("  1. Generate sample data: python generate_sample_data.py")
        print("  2. Run tests: python test_pipeline.py")
        print("  3. Run demo: python example.py")
        print("  4. Or train on your data: python train.py --data_path your_data.json")
        return True
    else:
        print("⚠️  Setup incomplete. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

