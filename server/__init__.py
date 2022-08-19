import os
import sys

parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_folder not in sys.path:
    sys.path.append(parent_folder)