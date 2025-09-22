"""
Test runner for AI-Powered Delivery Failure Analysis

This script runs all tests in the correct order.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def run_test(test_file):
    """Run a single test file."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print('='*60)
    
    try:
        # Change to project root directory
        project_root = Path(__file__).parent.parent
        test_file_path = project_root / 'tests' / test_file
        result = subprocess.run([sys.executable, str(test_file_path)], 
                              capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print(f"✅ {test_file} passed")
            print(result.stdout)
        else:
            print(f"❌ {test_file} failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False
    
    return True

def main():
    """Run all tests in order."""
    print("STARTING ALL TESTS")
    print("="*60)
    
    # Define test files in order
    test_files = [
        "test_task1.py",
        "test_task2.py", 
        "test_task3.py"
    ]
    
    passed = 0
    total = len(test_files)
    
    for test_file in test_files:
        if run_test(test_file):
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed}/{total} test suites passed")
    print('='*60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
