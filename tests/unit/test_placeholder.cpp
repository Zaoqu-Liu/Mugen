#include <cstdio>
#include <cstdlib>

#include "mugen/core/types.h"

#define MUGEN_CHECK(cond)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__,      \
                         __LINE__);                                           \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

static void test_dims_default() {
    mugen::Dims d{};
    MUGEN_CHECK(d.rows == 0);
    MUGEN_CHECK(d.cols == 0);
}

static void test_quant_type_values() {
    MUGEN_CHECK(static_cast<mugen::u32>(mugen::QuantType::F16)  == 0);
    MUGEN_CHECK(static_cast<mugen::u32>(mugen::QuantType::Q8_0) == 1);
    MUGEN_CHECK(static_cast<mugen::u32>(mugen::QuantType::Q4_0) == 2);
    MUGEN_CHECK(static_cast<mugen::u32>(mugen::QuantType::Q4_K) == 3);
}

int main() {
    test_dims_default();
    test_quant_type_values();
    std::printf("All tests passed.\n");
    return 0;
}
