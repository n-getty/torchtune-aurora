#!/bin/bash
echo "=== START ==="
echo "Original PATH head: $(echo $PATH | tr ':' '\n' | head -5)"
source /usr/share/lmod/lmod/init/bash
module load frameworks/2025.3.1 2>&1 | tail -3
echo "After module PATH head: $(echo $PATH | tr ':' '\n' | head -5)"
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
echo "After filter PATH head: $(echo $PATH | tr ':' '\n' | head -5)"
echo "which python3: $(which python3)"
