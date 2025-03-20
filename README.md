# ComparatorNetworks.jl

**Copyright © 2025 by David K. Zhang. Released under the [MIT License][1].**

[**Comparator networks**][2], also known as **sorting networks**, are fast algorithms for sorting or accumulating a fixed number of inputs. **ComparatorNetworks.jl** is a Julia package for discovering, analyzing, testing, and optimizing comparator networks.

![ComparatorNetwork](https://github.com/user-attachments/assets/ab6a3d13-5412-4405-88ad-bde92e549fde)

A comparator network consists of horizontal **wires** and an ordered sequence of vertical **comparators**. Each comparator connects two wires and represents an operation, such as [**`minmax`**][3] or [**TwoSum**][4], that takes and returns two values. Comparator networks are called **sorting networks** when used to sort their input data, but they can also be applied to other tasks, such as partial sorting, median selection, and [high-precision arithmetic][5].

Finding [optimal sorting networks][6] is a notoriously hard combinatorial optimization problem. The optimal number of comparators in a sorting network remains unknown as of 2025, even for as few as 13 inputs. **ComparatorNetworks.jl** implements heuristic optimization algorithms based on [simulated annealing][7] to find comparator networks of minimal length (number of comparators) and depth (number of sequential dependencies) for a variety of tasks.



## Usage Examples

**ComparatorNetworks.jl** is currently under active development. The function names and interfaces shown below may change in future releases.



[1]: https://github.com/dzhang314/ComparatorNetworks.jl/blob/main/LICENSE
[2]: https://en.wikipedia.org/wiki/Sorting_network
[3]: https://docs.julialang.org/en/v1/base/math/#Base.minmax
[4]: https://en.wikipedia.org/wiki/2Sum
[5]: https://github.com/dzhang314/MultiFloats.jl
[6]: https://bertdobbelaere.github.io/sorting_networks.html
[7]: https://en.wikipedia.org/wiki/Simulated_annealing
