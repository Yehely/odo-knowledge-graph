"""
CLI entry point: Excel → hetero_gnn/processed_hetero_graph.pt

Usage:
    conda run -n odo python3 hetero_gnn/run_preprocess.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hetero_gnn.preprocess import main

if __name__ == "__main__":
    main()
