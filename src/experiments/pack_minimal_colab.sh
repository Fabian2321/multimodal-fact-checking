#!/bin/bash
# Minimales Colab-Archiv bauen
# Nur relevante Dateien und Daten werden gepackt

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

# 1. Vorherigen Temp-Ordner löschen
rm -rf "$TMP_DIR"

# 2. Minimalstruktur anlegen
mkdir -p "$TMP_DIR/data/downloaded_fakeddit_images"
mkdir -p "$TMP_DIR/data/processed"
mkdir -p "$TMP_DIR/data/external_knowledge"

# 3. Hauptskripte und Konfig kopieren
cp "$SRC_DIR/run_llava_experiment.py" "$TMP_DIR/"
cp "$SRC_DIR/quick_test.py" "$TMP_DIR/"
cp "$SRC_DIR/colab_quick_setup.py" "$TMP_DIR/"
cp "$SRC_DIR/README_COLAB.md" "$TMP_DIR/"
cp "$SRC_DIR/llava_experiments.json" "$TMP_DIR/"
cp "$SRC_DIR/requirements_colab.txt" "$TMP_DIR/"

# 4. Testdaten kopieren
cp "$SRC_DIR/data/processed/test_balanced_pairs.csv" "$TMP_DIR/data/processed/"

# 5. Externe Knowledge kopieren
cp "$SRC_DIR/data/external_knowledge/"*.json "$TMP_DIR/data/external_knowledge/"

# 6. Bilder (hier: alle, ggf. anpassen für kleine Tests)
cp "$SRC_DIR/data/downloaded_fakeddit_images/"*.jpg "$TMP_DIR/data/downloaded_fakeddit_images/"

# 7. Optional: weitere kleine Dateien kopieren (z.B. test_queries.json, falls benötigt)
# cp "$SRC_DIR/data/test_queries.json" "$TMP_DIR/data/"

# 8. Archiv bauen
cd /tmp
rm -f "$ARCHIVE"

# Try to use --no-mac-metadata first, fallback to standard tar if not supported
if tar --help 2>&1 | grep -q "no-mac-metadata"; then
    tar --no-mac-metadata -czf "$ARCHIVE" mllm_colab_minimal 2>/dev/null || tar -czf "$ARCHIVE" mllm_colab_minimal
else
    tar -czf "$ARCHIVE" mllm_colab_minimal
fi

mv "$ARCHIVE" "$OLDPWD/"

# 9. Aufräumen
rm -rf "$TMP_DIR"

# 10. Fertig
cd "$OLDPWD"
echo "FERTIG: $ARCHIVE bereit für Upload in Colab!"
echo "Enthält:"
echo "- Verbesserte LLaVA Experiment-Skripte"
echo "- Quick Test für Setup-Verifikation"
echo "- Alle Testdaten und Bilder"
echo "- Requirements-Datei" 