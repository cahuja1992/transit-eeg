#!/usr/bin/env python3
"""
Test runner script for TRANSIT-EEG.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --fast       # Run only fast tests
    python run_tests.py --unit       # Run only unit tests
    python run_tests.py --coverage   # Run with coverage report
"""

import sys
import argparse
import subprocess
from pathlib import Path


def run_tests(args):
    """Run pytest with specified arguments."""
    cmd = ['pytest']
    
    # Add test directory
    cmd.append('tests/')
    
    # Add verbosity
    if args.verbose:
        cmd.append('-vv')
    else:
        cmd.append('-v')
    
    # Filter by test type
    if args.fast:
        cmd.extend(['-m', 'not slow'])
    elif args.unit:
        cmd.extend(['-m', 'unit'])
    elif args.integration:
        cmd.extend(['-m', 'integration'])
    
    # Add coverage
    if args.coverage:
        cmd.extend([
            '--cov=src/transit_eeg',
            '--cov-report=term-missing',
            '--cov-report=html'
        ])
    
    # Add specific test file if provided
    if args.file:
        cmd = ['pytest', args.file, '-v']
    
    # Run pytest
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Run TRANSIT-EEG test suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all tests
  %(prog)s --fast             # Skip slow tests
  %(prog)s --unit             # Run only unit tests
  %(prog)s --coverage         # Generate coverage report
  %(prog)s --file tests/test_idpm.py  # Run specific test file
        """
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Skip slow tests'
    )
    
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run only unit tests'
    )
    
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run only integration tests'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Run specific test file'
    )
    
    args = parser.parse_args()
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"Using pytest version: {pytest.__version__}")
    except ImportError:
        print("Error: pytest is not installed.")
        print("Install it with: pip install pytest pytest-cov")
        return 1
    
    # Run tests
    return_code = run_tests(args)
    
    if return_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {return_code}")
    
    return return_code


if __name__ == '__main__':
    sys.exit(main())
