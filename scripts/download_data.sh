#!/usr/bin/env bash
# Clone the BCCD dataset into data/BCCD_Dataset. Idempotent: skips if already present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/BCCD_Dataset"
REPO_URL="https://github.com/Shenggan/BCCD_Dataset.git"

mkdir -p "$ROOT/data"
if [ -d "$DEST/BCCD/Annotations" ] && [ -d "$DEST/BCCD/JPEGImages" ]; then
    echo "BCCD already present at $DEST, skipping clone."
else
    rm -rf "$DEST"
    git clone --depth 1 "$REPO_URL" "$DEST"
fi

echo "JPEG count: $(find "$DEST/BCCD/JPEGImages" -name '*.jpg' | wc -l)"
echo "XML count:  $(find "$DEST/BCCD/Annotations" -name '*.xml' | wc -l)"
