# ComparatorNetworks.jl

**Copyright © 2025 by David K. Zhang. Released under the [MIT License][1].**

[**Comparator networks**][2], also known as **sorting networks**, are fast algorithms for sorting or accumulating a fixed number of inputs. **ComparatorNetworks.jl** is a Julia package for discovering, analyzing, testing, and optimizing comparator networks.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="-36 16 276 168" width="276" height="168">
    <line x1="0" y1="28" x2="0" y2="92" stroke="white" stroke-width="3" />
    <line x1="0" y1="108" x2="0" y2="172" stroke="white" stroke-width="3" />
    <line x1="60" y1="28" x2="60" y2="132" stroke="white" stroke-width="3" />
    <line x1="84" y1="68" x2="84" y2="172" stroke="white" stroke-width="3" />
    <line x1="144" y1="68" x2="144" y2="132" stroke="white" stroke-width="3" />
    <line x1="204" y1="28" x2="204" y2="92" stroke="white" stroke-width="3" />
    <line x1="-36" y1="40" x2="240" y2="40" stroke="white" stroke-width="3" />
    <line x1="-36" y1="80" x2="240" y2="80" stroke="white" stroke-width="3" />
    <line x1="-36" y1="120" x2="240" y2="120" stroke="white" stroke-width="3" />
    <line x1="-36" y1="160" x2="240" y2="160" stroke="white" stroke-width="3" />
    <line x1="0" y1="28" x2="0" y2="92" stroke="black" stroke-width="1" />
    <line x1="0" y1="108" x2="0" y2="172" stroke="black" stroke-width="1" />
    <line x1="60" y1="28" x2="60" y2="132" stroke="black" stroke-width="1" />
    <line x1="84" y1="68" x2="84" y2="172" stroke="black" stroke-width="1" />
    <line x1="144" y1="68" x2="144" y2="132" stroke="black" stroke-width="1" />
    <line x1="204" y1="28" x2="204" y2="92" stroke="black" stroke-width="1" />
    <line x1="-36" y1="40" x2="240" y2="40" stroke="black" stroke-width="1" />
    <line x1="-36" y1="80" x2="240" y2="80" stroke="black" stroke-width="1" />
    <line x1="-36" y1="120" x2="240" y2="120" stroke="black" stroke-width="1" />
    <line x1="-36" y1="160" x2="240" y2="160" stroke="black" stroke-width="1" />
    <circle cx="0" cy="40" r="12" stroke="white" stroke-width="3" fill="none" />
    <circle cx="0" cy="80" r="12" stroke="white" stroke-width="3" fill="none" />
    <circle cx="0" cy="120" r="12" stroke="white" stroke-width="3" fill="none" />
    <circle cx="0" cy="160" r="12" stroke="white" stroke-width="3" fill="none" />
    <circle cx="60" cy="40" r="12" stroke="white" stroke-width="3" fill="none" />
    <circle cx="60" cy="120" r="12" stroke="white" stroke-width="3" fill="none" />
    <circle cx="84" cy="80" r="12" stroke="white" stroke-width="3" fill="none" />
    <circle cx="84" cy="160" r="12" stroke="white" stroke-width="3" fill="none" />
    <circle cx="144" cy="80" r="12" stroke="white" stroke-width="3" fill="none" />
    <circle cx="144" cy="120" r="12" stroke="white" stroke-width="3" fill="none" />
    <circle cx="204" cy="40" r="12" stroke="white" stroke-width="3" fill="none" />
    <circle cx="204" cy="80" r="12" stroke="white" stroke-width="3" fill="none" />
    <circle cx="0" cy="40" r="12" stroke="black" stroke-width="1" fill="none" />
    <circle cx="0" cy="80" r="12" stroke="black" stroke-width="1" fill="none" />
    <circle cx="0" cy="120" r="12" stroke="black" stroke-width="1" fill="none" />
    <circle cx="0" cy="160" r="12" stroke="black" stroke-width="1" fill="none" />
    <circle cx="60" cy="40" r="12" stroke="black" stroke-width="1" fill="none" />
    <circle cx="60" cy="120" r="12" stroke="black" stroke-width="1" fill="none" />
    <circle cx="84" cy="80" r="12" stroke="black" stroke-width="1" fill="none" />
    <circle cx="84" cy="160" r="12" stroke="black" stroke-width="1" fill="none" />
    <circle cx="144" cy="80" r="12" stroke="black" stroke-width="1" fill="none" />
    <circle cx="144" cy="120" r="12" stroke="black" stroke-width="1" fill="none" />
    <circle cx="204" cy="40" r="12" stroke="black" stroke-width="1" fill="none" />
    <circle cx="204" cy="80" r="12" stroke="black" stroke-width="1" fill="none" />
</svg>

A comparator network consists of horizontal **wires** and an ordered sequence of vertical **comparators**. Each comparator connects two wires and represents an operation, such as [`minmax`][3] or [**TwoSum**][4], that takes and returns two values.

Comparator networks are called **sorting networks** when used to sort their input data, but they can also be applied to other tasks, such as partial sorting, median/quartile selection, and [high-precision arithmetic][5].

Finding [optimal sorting networks][6] is a notoriously hard combinatorial optimization problem. As of 2025, it is an open problem to determine the optimal number of comparators in a sorting network, even for as few as 13 inputs.

[1]: https://github.com/dzhang314/ComparatorNetworks.jl/blob/main/LICENSE
[2]: https://en.wikipedia.org/wiki/Sorting_network
[3]: https://docs.julialang.org/en/v1/base/math/#Base.minmax
[4]: https://en.wikipedia.org/wiki/2Sum
[5]: https://github.com/dzhang314/MultiFloats.jl
[6]: https://bertdobbelaere.github.io/sorting_networks.html
