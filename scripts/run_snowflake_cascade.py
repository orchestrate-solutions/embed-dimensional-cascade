#!/usr/bin/env python3
"""
Runner script for Snowflake Dimensional Cascade

This script provides a simple CLI wrapper around the Snowflake Cascade implementation,
allowing easy access to all features of the Dimensional Cascade approach.
"""
import sys
import os

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import main module from the package
try:
    from dimensional_cascade.snowflake_cascade import main
except ImportError:
    print("Error: Could not import Snowflake Cascade module.")
    print("Make sure the dimensional_cascade package is properly installed.")
    sys.exit(1)

if __name__ == "__main__":
    main() 