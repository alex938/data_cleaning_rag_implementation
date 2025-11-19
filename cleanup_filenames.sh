#!/bin/bash
# Cleanup script to remove problematic characters from filenames and directories
# This removes carriage returns (\r), newlines, and other invisible characters
# Usage: bash cleanup_filenames.sh [directory]

TARGET_DIR="${1:-./data}"

echo "🧹 Cleaning up filenames in: $TARGET_DIR"
echo ""

# Counter for renamed items
RENAMED_COUNT=0

# Function to clean a filename
clean_filename() {
    local original="$1"
    # Remove carriage returns, newlines, tabs, and other control characters
    # tr -d removes the specified characters
    local cleaned=$(echo "$original" | tr -d '\r\n\t\000-\037')
    echo "$cleaned"
}

# Find all files and directories, process from deepest to shallowest
# (so we rename files before their parent directories)
find "$TARGET_DIR" -depth -print0 | while IFS= read -r -d '' item; do
    # Get the directory and basename
    dir=$(dirname "$item")
    base=$(basename "$item")
    
    # Clean the basename
    cleaned_base=$(clean_filename "$base")
    
    # Check if name needs cleaning
    if [ "$base" != "$cleaned_base" ]; then
        new_path="$dir/$cleaned_base"
        
        echo "📝 Renaming:"
        echo "   From: $item"
        echo "   To:   $new_path"
        
        # Check if target already exists
        if [ -e "$new_path" ]; then
            echo "   ⚠️  WARNING: Target already exists, skipping..."
        else
            mv "$item" "$new_path"
            if [ $? -eq 0 ]; then
                echo "   ✅ Renamed successfully"
                ((RENAMED_COUNT++))
            else
                echo "   ❌ Failed to rename"
            fi
        fi
        echo ""
    fi
done

echo ""
echo "======================================"
echo "✅ Cleanup complete!"
echo "   Renamed items: $RENAMED_COUNT"
echo "======================================"

# List any remaining issues
echo ""
echo "Checking for remaining problematic names..."
if find "$TARGET_DIR" -name $'*\r*' -o -name $'*\n*' -o -name $'*\t*' 2>/dev/null | grep -q .; then
    echo "⚠️  Found items that still need attention:"
    find "$TARGET_DIR" -name $'*\r*' -o -name $'*\n*' -o -name $'*\t*' 2>/dev/null
else
    echo "✅ No problematic filenames found!"
fi
