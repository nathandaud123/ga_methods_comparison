"""
Script to download Solomon benchmark datasets
Note: This is a helper script. You may need to download datasets manually.
"""

import os
import urllib.request
from pathlib import Path

# Solomon benchmark URLs (these may need to be updated)
SOLOMON_URLS = {
    'C101': 'http://web.cba.neu.edu/~msolomon/problems.htm',
    'C201': 'http://web.cba.neu.edu/~msolomon/problems.htm',
    'R101': 'http://web.cba.neu.edu/~msolomon/problems.htm',
    'R201': 'http://web.cba.neu.edu/~msolomon/problems.htm',
    'RC101': 'http://web.cba.neu.edu/~msolomon/problems.htm',
    'RC201': 'http://web.cba.neu.edu/~msolomon/problems.htm',
}

def download_solomon_datasets():
    """Download Solomon benchmark datasets"""
    data_dir = Path('data/solomon')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Solomon benchmark datasets need to be downloaded manually.")
    print("Please visit: http://web.cba.neu.edu/~msolomon/problems.htm")
    print(f"Place the .txt files in: {data_dir.absolute()}")
    print("\nRequired files:")
    for filename in ['C101.txt', 'C201.txt', 'R101.txt', 'R201.txt', 'RC101.txt', 'RC201.txt']:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (missing)")

if __name__ == '__main__':
    download_solomon_datasets()

