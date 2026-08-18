#!/usr/bin/env python3
"""Convenience wrapper to run a single jax-ehl simulation."""

import sys

from gpu_ehl.cli import main

if __name__ == "__main__":
    sys.exit(main())
