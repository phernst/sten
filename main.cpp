#include "sten.h"

int main()
{
    using namespace sten::indexing;
    const auto t = sten::arange(16);
    const auto u = t.index({{1, 14, 2}});
    const auto v = u.index({{1, None, 2}});
    t.print();
    u.print();
    v.print();
}
