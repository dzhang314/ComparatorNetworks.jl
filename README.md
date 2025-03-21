# ComparatorNetworks.jl

**Copyright © 2025 by David K. Zhang. Released under the [MIT License][1].**

[**Comparator networks**][2], also known as **sorting networks**, are fast algorithms for sorting or accumulating a fixed number of inputs. **ComparatorNetworks.jl** is a Julia package for discovering, analyzing, testing, and optimizing comparator networks.

![ComparatorNetwork](https://github.com/user-attachments/assets/ab6a3d13-5412-4405-88ad-bde92e549fde)

A comparator network consists of horizontal **wires** and an ordered sequence of vertical **comparators**. Each comparator connects two wires and represents an operation, such as [**`minmax`**][3] or [**TwoSum**][4], that takes and returns two values. Comparator networks are called **sorting networks** when used to sort their input data, but they can also be applied to other tasks, such as partial sorting, median selection, and [high-precision arithmetic][5].

Finding [optimal sorting networks][6] is a notoriously hard combinatorial optimization problem. The optimal number of comparators in a sorting network remains unknown as of 2025, even for as few as 13 inputs. **ComparatorNetworks.jl** implements heuristic optimization algorithms based on [simulated annealing][7] to find comparator networks of minimal length (number of comparators) and depth (number of sequential dependencies) for a variety of tasks.



## Usage Examples

> **Note:** **ComparatorNetworks.jl** is currently under active development. The function names and interfaces shown below may change in future releases.

Suppose we want to find a comparator network that sorts every possible permutation of 3 inputs. We first set up a _condition object_ to represent the desired condition. Then, we call `generate_comparator_network(condition)` to find a random comparator network that satisfies the condition.

```
using ComparatorNetworks

condition = PassesAllTests(
    [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)],
    minmax, issorted)
network = generate_comparator_network(condition)
display(network)
```

This will return one of several possible outputs:

> ![Example1](https://github.com/user-attachments/assets/f7fc4213-92fa-49e3-8c42-d0534406c00c) or
> ![Example2](https://github.com/user-attachments/assets/0fbd60bb-48a0-47e9-9839-a00fb818614d) or
> ![Example3](https://github.com/user-attachments/assets/07079fc8-c7aa-4be2-9cfc-97d41b1af7c5) or
> ![Example4](https://github.com/user-attachments/assets/bb267c5c-46c3-47ca-86f2-963c3b1b10eb)

In the Julia REPL, this will display in textual form, e.g., `ComparatorNetwork{3}(Tuple{UInt8, UInt8}[(0x02, 0x03), (0x01, 0x03), (0x01, 0x02)])`. In Jupyter, this will display as a graphical SVG file, as shown above.



[1]: https://github.com/dzhang314/ComparatorNetworks.jl/blob/main/LICENSE
[2]: https://en.wikipedia.org/wiki/Sorting_network
[3]: https://docs.julialang.org/en/v1/base/math/#Base.minmax
[4]: https://en.wikipedia.org/wiki/2Sum
[5]: https://github.com/dzhang314/MultiFloats.jl
[6]: https://bertdobbelaere.github.io/sorting_networks.html
[7]: https://en.wikipedia.org/wiki/Simulated_annealing
