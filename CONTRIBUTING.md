# Contributing to Mugen

Thank you for your interest in contributing! Mugen is a MoE extreme inference engine for Apple Silicon. This document outlines how to contribute effectively.

## Code of Conduct

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs
- Use the [Bug Report](https://github.com/Zaoqu-Liu/Mugen/issues/new?template=bug_report.md) template
- Include your macOS version, hardware specs, Mugen version, and steps to reproduce
- Attach relevant logs or error messages

### Requesting Features
- Use the [Feature Request](https://github.com/Zaoqu-Liu/Mugen/issues/new?template=feature_request.md) template
- Describe the problem your feature solves and how it benefits the community

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes
4. Run `clang-format` on modified C++ files
5. Build with `-Werror`: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`
6. Run tests: `ctest --test-dir build --output-on-failure`
7. Commit with a descriptive message
8. Push and open a PR against `main`

## Development Setup

```sh
# Clone
git clone https://github.com/Zaoqu-Liu/Mugen.git
cd Mugen

# System check
python3 tools/check_system.py

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)

# Run tests
ctest --test-dir build --output-on-failure
```

## Coding Style

- **C++23** with `-Werror` in Release mode
- **Zero warnings** — treat warnings as errors
- Format with `.clang-format` before committing
- Prefer `std::expected<T, std::string>` for error handling (no exceptions)
- Metal kernels: add to `kXxxSource` individually and append to `kAllKernelsSource`
- No third-party dependencies

## Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `perf:` — performance improvement
- `refactor:` — code restructuring
- `test:` — adding or updating tests
- `docs:` — documentation changes
- `build:` — build system or CI changes

## Testing

- Write unit tests for new features
- Tests go in `tests/unit/` and are auto-discovered by CMake
- `ctest --test-dir build --output-on-failure` to verify

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
