#!/bin/bash
set -euo pipefail

# Format all C++ source files with clang-format
find src include tests -name '*.cpp' -o -name '*.h' -o -name '*.mm' | xargs clang-format -i
echo "Formatted $(find src include tests -name '*.cpp' -o -name '*.h' -o -name '*.mm' | wc -l) files"
