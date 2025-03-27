module ComparatorNetworks

using Random: shuffle
using SIMD: Vec

include("Comparators.jl")
using .Comparators
export bitminmax, two_sum, annihilating_maxmin
export forward_pass, forward_fixed_point,
    backward_pass, backward_fixed_point,
    riffle_pass, riffle_fixed_point
export isbitsorted, IsCoarselySorted,
    IsNormalized, RelativeErrorBound, IsCorrectlyRounded

include("TestCaseGenerators.jl")
using .TestCaseGenerators
export all_bit_vectors
export coarse_mfadd_test_cases
export rand_vec_u64, rand_vec_f64, rand_vec_mf64
export prepare_mfadd_inputs, prepare_mfmul_inputs

############################################## COMPARATOR NETWORK DATA STRUCTURE


export ComparatorNetwork


struct ComparatorNetwork{N}
    comparators::Vector{Tuple{UInt8,UInt8}}

    function ComparatorNetwork{N}(comparators::AbstractVector) where {N}
        @assert 0 <= N <= typemax(UInt8)
        my_comparators = Tuple{UInt8,UInt8}[]
        sizehint!(my_comparators, length(comparators))
        for (a, b) in comparators
            my_a = UInt8(a)
            @assert 0 < my_a <= N
            my_b = UInt8(b)
            @assert 0 < my_b <= N
            if my_a != my_b
                push!(my_comparators, minmax(my_a, my_b))
            end
        end
        return new{N}(my_comparators)
    end
end


@inline Base.length(network::ComparatorNetwork{N}) where {N} =
    length(network.comparators)
@inline Base.:(==)(a::ComparatorNetwork{N}, b::ComparatorNetwork{N}) where {N} =
    (a.comparators == b.comparators)
@inline Base.copy(network::ComparatorNetwork{N}) where {N} =
    ComparatorNetwork{N}(network.comparators)
@inline Base.hash(network::ComparatorNetwork{N}, h::UInt) where {N} =
    hash(network.comparators, hash(N, h))


function Base.isless(
    a::ComparatorNetwork{M},
    b::ComparatorNetwork{N},
) where {M,N}
    if M != N
        return isless(M, N)
    end
    len_a = length(a.comparators)
    len_b = length(b.comparators)
    if len_a != len_b
        return isless(len_a, len_b)
    end
    return isless(a.comparators, b.comparators)
end


include("Output.jl")
using .Output
export hexfloat
export svg_string


############################################################## DEPTH COMPUTATION


export depth, canonize


@inline function depth(network::ComparatorNetwork{N}) where {N}
    generation = ntuple(_ -> 0, Val{N}())
    for (i, j) in network.comparators
        @assert 1 <= i < j <= N
        gi = @inbounds generation[i]
        gj = @inbounds generation[j]
        age = max(gi, gj) + 1
        generation = @inbounds Base.setindex(generation, age, i)
        generation = @inbounds Base.setindex(generation, age, j)
    end
    return maximum(generation)
end


function canonize(network::ComparatorNetwork{N}) where {N}
    if isempty(network.comparators)
        return ComparatorNetwork{N}(Tuple{UInt8,UInt8}[])
    end
    last_compared = ntuple(_ -> zero(UInt8), Val{N}())
    generation = ntuple(_ -> 0, Val{N}())
    comparators = Vector{Tuple{UInt8,UInt8}}[]
    for (i, j) in network.comparators
        @assert 1 <= i < j <= N
        lci = @inbounds last_compared[i]
        lcj = @inbounds last_compared[j]
        if (lci != j) | (lcj != i)
            gi = @inbounds generation[i]
            gj = @inbounds generation[j]
            age = max(gi, gj) + 1
            if age > length(comparators)
                @assert age == length(comparators) + 1
                push!(comparators, [(i, j)])
            else
                push!((@inbounds comparators[age]), (i, j))
            end
            last_compared = @inbounds Base.setindex(last_compared, j, i)
            last_compared = @inbounds Base.setindex(last_compared, i, j)
            generation = @inbounds Base.setindex(generation, age, i)
            generation = @inbounds Base.setindex(generation, age, j)
        end
    end
    for v in comparators
        sort!(v)
    end
    return ComparatorNetwork{N}(reduce(vcat, comparators))
end


################################################### COMPARATOR NETWORK EXECUTION


export run_comparator_network!, run_comparator_network


@inline function _unsafe_run_comparator_network!(
    data::AbstractVector{T},
    network::ComparatorNetwork{N},
    comparator::C,
) where {N,T,C}
    # Assumes: data has indices 1:N.
    # Assumes: network is well-formed (comparator indices lie in 1:N).
    for (i, j) in network.comparators
        @inbounds @inline data[i], data[j] = comparator(data[i], data[j])
    end
    return data
end


@inline function run_comparator_network!(
    data::AbstractVector{T},
    network::ComparatorNetwork{N},
    comparator::C,
) where {N,T,C}
    # Assumes: network is well-formed (comparator indices lie in 1:N).
    Base.require_one_based_indexing(data)
    @assert length(data) == N
    return _unsafe_run_comparator_network!(data, network, comparator)
end


function run_comparator_network(
    input::NTuple{N,T},
    network::ComparatorNetwork{N},
    comparator::C,
) where {N,T,C}
    # Assumes: network is well-formed (comparator indices lie in 1:N).
    return _unsafe_run_comparator_network!(collect(input), network, comparator)
end


############################################# COMPARATOR NETWORK CODE GENERATION


export generate_code


function generate_code(network::ComparatorNetwork{N}, comparator) where {N}
    xs = [Symbol('x', i) for i in Base.OneTo(N)]
    body = Expr[]
    push!(body, Expr(:meta, :inline))
    push!(body, Expr(:(=), Expr(:tuple, xs...), :x))
    for (i, j) in network.comparators
        @assert 1 <= i < j <= N
        push!(body, Expr(:(=), Expr(:tuple, xs[i], xs[j]),
            Expr(:call, comparator, xs[i], xs[j])))
    end
    push!(body, Expr(:return, Expr(:tuple, xs...)))
    return Expr(:->, :x, Expr(:block, body...))
end


################################################## COMPARATOR NETWORK CONDITIONS


export AbstractCondition, PassesTest, PassesAllTests


abstract type AbstractCondition{N} end


struct PassesTest{N,T,C,P} <: AbstractCondition{N}
    test_case::NTuple{N,T}
    comparator::C
    postcondition::P
    buffer::Vector{T}
end


struct PassesAllTests{N,T,C,P} <: AbstractCondition{N}
    test_cases::Vector{NTuple{N,T}}
    comparator::C
    postcondition::P
    buffer::Vector{T}
end


function PassesTest(
    test_case::NTuple{N,T},
    comparator::C,
    postcondition::P,
) where {N,T,C,P}
    buffer = Vector{T}(undef, N)
    return PassesTest{N,T,C,P}(
        test_case, comparator, postcondition, buffer)
end


function PassesAllTests(
    test_cases::Vector{NTuple{N,T}},
    comparator::C,
    postcondition::P,
) where {N,T,C,P}
    buffer = Vector{T}(undef, N)
    return PassesAllTests{N,T,C,P}(
        test_cases, comparator, postcondition, buffer)
end


@inline function (tester::PassesTest{N,T,C,P})(
    network::ComparatorNetwork{N},
) where {N,T,C,P}
    # Assumes: network is well-formed (comparator indices lie in 1:N).
    @simd ivdep for i = 1:N
        @inbounds tester.buffer[i] = tester.test_case[i]
    end
    _unsafe_run_comparator_network!(
        tester.buffer, network, tester.comparator)
    return tester.postcondition(tester.buffer)
end


@inline function (tester::PassesAllTests{N,T,C,P})(
    network::ComparatorNetwork{N},
) where {N,T,C,P}
    # Assumes: network is well-formed (comparator indices lie in 1:N).
    for test_case in tester.test_cases
        @simd ivdep for i = 1:N
            @inbounds tester.buffer[i] = test_case[i]
        end
        _unsafe_run_comparator_network!(
            tester.buffer, network, tester.comparator)
        if !tester.postcondition(tester.buffer)
            return false
        end
    end
    return true
end


################################################## COMPARATOR NETWORK GENERATION


export prune!, generate_comparator_network


@inline function _test_conditions(
    network::ComparatorNetwork{N},
    conditions::AbstractCondition{N}...,
) where {N}
    for condition in conditions
        if !condition(network)
            return false
        end
    end
    return true
end


function prune!(
    network::ComparatorNetwork{N},
    conditions::AbstractCondition{N}...,
) where {N}
    while true
        found = false
        for i in shuffle(1:length(network.comparators))
            original_comparator = network.comparators[i]
            deleteat!(network.comparators, i)
            if _test_conditions(network, conditions...)
                found = true
                break
            else
                insert!(network.comparators, i, original_comparator)
            end
        end
        if !found
            return network
        end
    end
end


@inline function _random_comparator(::Val{N}) where {N}
    i = rand(0x01:UInt8(N))
    j = rand(0x01:UInt8(N - 1))
    j += UInt8(j >= i)
    return minmax(i, j)
end


function generate_comparator_network(
    conditions::AbstractCondition{N}...,
) where {N}
    network = ComparatorNetwork{N}(Tuple{UInt8,UInt8}[])
    while !_test_conditions(network, conditions...)
        push!(network.comparators, _random_comparator(Val{N}()))
    end
    return prune!(network, conditions...)
end


################################################ COMPARATOR NETWORK OPTIMIZATION


include("Annealing.jl")
using .Annealing
export mutate_comparator_network!, pareto_dominates, pareto_push!,
    optimize_comparator_network


##################################################### COMBINATORIAL OPTIMIZATION


export greedy_hitting_set


function greedy_hitting_set(sets::AbstractVector{<:AbstractSet{T}}) where {T}
    # Given sets S_1, ..., S_n, select items x_1, ..., x_m from those sets
    # such that each set S_i contains at least one selected item x_j.
    # It is NP-complete to find the optimal selection that minimizes m.
    # This function implements a greedy approximation strategy.
    @assert isbitstype(T)
    hit_sets = Dict{T,BitSet}()
    for (i, set) in enumerate(sets)
        for item in set
            if !haskey(hit_sets, item)
                hit_sets[item] = BitSet()
            end
            push!(hit_sets[item], i)
        end
    end
    result = T[]
    while !isempty(hit_sets)
        count, item = findmax(length, hit_sets)
        if iszero(count)
            break
        end
        push!(result, item)
        for index in hit_sets[item]
            for other_item in sets[index]
                if other_item !== item
                    delete!(hit_sets[other_item], index)
                end
            end
        end
        delete!(hit_sets, item)
    end
    return result
end


################################################################################

end # module ComparatorNetworks
