"""CLI wrapper; importing this module never reads or writes a dataset."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from l20.export import main

if __name__=='__main__':
    main()
