#!/usr/bin/env bash
# merge_py.sh  <directory> [file_output]

# Directory da scandagliare (default: directory corrente)
DIR=${1:-.}

# File di output (default: merged_py_<nome-directory>.txt)
OUT=${2:-merged_py_$(basename "$DIR").txt}

# Svuota/crea il file di output
: > "$OUT"

# Cerca soltanto i .py (case-insensitive) ed evita di includere l'output stesso
find "$DIR" -type f -iname '*.py' ! -path "$OUT" -print0 |
while IFS= read -r -d '' FILE; do
    printf '\n===== %s =====\n' "$FILE" >> "$OUT"
    cat "$FILE" >> "$OUT"
done

echo "File unificato creato in: $OUT"
