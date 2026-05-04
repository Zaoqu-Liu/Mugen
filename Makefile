BUILD_DIR  := build
DEBUG_DIR  := build-debug
CMAKE      := cmake
NPROC      := $(shell sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

.PHONY: build debug test bench clean format check doctor serve help

## build     : Release build
build:
	@$(CMAKE) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	@$(CMAKE) --build $(BUILD_DIR) -j$(NPROC)

## debug     : Debug build with sanitizers
debug:
	@$(CMAKE) -B $(DEBUG_DIR) -DCMAKE_BUILD_TYPE=Debug -DMUGEN_ENABLE_ASAN=ON
	@$(CMAKE) --build $(DEBUG_DIR) -j$(NPROC)

## test      : Run unit tests
test: build
	@cd $(BUILD_DIR) && ctest --output-on-failure -j$(NPROC)

## bench     : Run benchmarks
bench: build
	@cd $(BUILD_DIR) && ctest -R bench --output-on-failure

## clean     : Remove build artifacts
clean:
	@rm -rf $(BUILD_DIR) $(DEBUG_DIR)

## format    : Format source with clang-format
format:
	@find src include tests bench -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.mm' -o -name '*.metal' \
		| xargs clang-format -i
	@echo "Formatted."

## check     : Verify formatting (CI-friendly)
check:
	@find src include tests bench -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.mm' -o -name '*.metal' \
		| xargs clang-format --dry-run --Werror

## doctor    : Check system requirements
doctor:
	@python3 tools/check_system.py

## serve     : Build and start the server
serve: build
	@$(BUILD_DIR)/src/server/mugen-server

## help      : Show this help
help:
	@grep '^##' $(MAKEFILE_LIST) | sed 's/## //' | column -t -s ':'
