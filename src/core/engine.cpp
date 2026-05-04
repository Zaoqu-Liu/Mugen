#include "mugen/mugen.h"

namespace mugen {
namespace detail {

static_assert(sizeof(f16) == 2, "f16 must be 2 bytes");
static_assert(sizeof(f32) == 4, "f32 must be 4 bytes");

}  // namespace detail
}  // namespace mugen
