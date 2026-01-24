#!/usr/bin/env python3
"""
Monitor Q4 2025 Validation Progress
====================================

Monitors the validation run and provides progress updates.
"""

import time
import re
from pathlib import Path

def get_progress():
    """Extract progress from log file."""
    log_file = Path('logs/q4_validation_run.log')
    
    if not log_file.exists():
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find last progress line
    progress_lines = re.findall(r'Processing:\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)', content)
    
    if progress_lines:
        last = progress_lines[-1]
        pct = int(last[0])
        current = int(last[1])
        total = int(last[2])
        return {'percent': pct, 'current': current, 'total': total}
    
    return None

def main():
    """Monitor progress and display updates."""
    print("Monitoring Q4 2025 Validation Progress")
    print("=" * 60)
    print()
    
    last_current = 0
    
    while True:
        progress = get_progress()
        
        if progress:
            if progress['current'] != last_current:
                elapsed_pct = progress['percent']
                remaining_pct = 100 - elapsed_pct
                
                # Estimate time
                if elapsed_pct > 0:
                    # Assume ~5 seconds per fixture
                    remaining_fixtures = progress['total'] - progress['current']
                    eta_seconds = remaining_fixtures * 5
                    eta_minutes = eta_seconds / 60
                    
                    print(f"Progress: {progress['current']}/{progress['total']} ({elapsed_pct}%)")
                    print(f"ETA: ~{eta_minutes:.0f} minutes")
                    print()
                
                last_current = progress['current']
                
                # Check if complete
                if progress['current'] >= progress['total']:
                    print("✅ Validation complete!")
                    print()
                    print("Results will be in:")
                    print("  - logs/q4_validation_run.log (full log)")
                    print("  - data/predictions/q4_2025_improved_validation_*.csv (results)")
                    break
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == '__main__':
    main()
