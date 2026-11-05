"""Same mapping as data_for_RL; add --show-shadow for overlay."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from l20.preview import main

if __name__=='__main__':
    main()
