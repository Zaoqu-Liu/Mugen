#!/bin/bash
set -euo pipefail

# Check formatting without modifying files
ERRORS=0
for f in $(find src include tests -name '*.cpp' -o -name '*.h' -o -name '*.mm'); do
    if ! clang-format --dry-run --Werror "$f" 2>/dev/null; then
        echo "FAIL: $f"
        ERRORS=$((ERRORS + 1))
    fi
done
if [ $ERRORS -eq 0 ]; then
    echo "All files pass format check"
else
    echo "$ERRORS files need formatting. Run: make format"
    exit 1
fi
