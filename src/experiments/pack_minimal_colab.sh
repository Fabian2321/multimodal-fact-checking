#!/bin/bash
# Build minimal Colab archive
# Only relevant files and data are packed

set -e

# Disable job control to prevent fg/bg errors in non-interactive shells
set +m

# Ensure we're in a proper shell environment
if [ -z "$BASH_VERSION" ]; then
    echo "Error: This script requires bash"
    exit 1
fi

SRC_DIR="drive_export/mllm_colab"
TMP_DIR="/tmp/mllm_colab_minimal"
ARCHIVE="mllm_colab_minimal.tar.gz"

# 1. Delete previous temp directory
rm -rf "$TMP_DIR"

# 2. Create minimal structure
mkdir -p "$TMP_DIR/data/downloaded_fakeddit_images"
mkdir -p "$TMP_DIR/data/processed"
mkdir -p "$TMP_DIR/data/external_knowledge"

# 3. Copy main scripts and config
cp "$SRC_DIR/run_llava_experiment.py" "$TMP_DIR/"
cp "$SRC_DIR/quick_test.py" "$TMP_DIR/"
cp "$SRC_DIR/colab_quick_setup.py" "$TMP_DIR/"
cp "$SRC_DIR/README_COLAB.md" "$TMP_DIR/"
cp "$SRC_DIR/llava_experiments.json" "$TMP_DIR/"
cp "$SRC_DIR/requirements_colab.txt" "$TMP_DIR/"

# 4. Copy test data
cp "$SRC_DIR/data/processed/test_balanced_pairs.csv" "$TMP_DIR/data/processed/"

# 5. Copy external knowledge
cp "$SRC_DIR/data/external_knowledge/"*.json "$TMP_DIR/data/external_knowledge/"

# 6. Images (here: all, adjust for small tests if needed)
cp "$SRC_DIR/data/downloaded_fakeddit_images/"*.jpg "$TMP_DIR/data/downloaded_fakeddit_images/"

# 7. Optional: copy additional small files (e.g., test_queries.json if needed)
# cp "$SRC_DIR/data/test_queries.json" "$TMP_DIR/data/"

# 8. Build archive
cd /tmp
rm -f "$ARCHIVE"

# Try to use --no-mac-metadata first, fallback to standard tar if not supported
if tar --help 2>&1 | grep -q "no-mac-metadata"; then
    tar --no-mac-metadata -czf "$ARCHIVE" mllm_colab_minimal 2>/dev/null || tar -czf "$ARCHIVE" mllm_colab_minimal
else
    tar -czf "$ARCHIVE" mllm_colab_minimal
fi

mv "$ARCHIVE" "$OLDPWD/"

# 9. Cleanup
rm -rf "$TMP_DIR"

# 10. Done
cd "$OLDPWD"
echo "DONE: $ARCHIVE ready for upload to Colab!"
echo "Contains:"
echo "- Improved LLaVA experiment scripts"
echo "- Quick test for setup verification"
echo "- All test data and images"
echo "- Requirements file" 