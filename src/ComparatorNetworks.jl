module ComparatorNetworks

using Random: shuffle
using SIMD: Vec

include("Comparators.jl")
using .Comparators
export bitminmax, two_sum, annihilating_maxmin
export top_down_bitbubble, top_down_accumulate, top_down_bitsort,
    top_down_normalize, bottom_up_bitbubble, bottom_up_accumulate,
    bottom_up_bitsort, bottom_up_normalize, alternating_bitbubble,
    alternating_accumulate, alternating_bitsort, alternating_normalize
export isbitsorted

include("TestCaseGenerators.jl")
using .TestCaseGenerators
export all_bit_vectors
export coarse_mfadd_test_cases
export rand_vec_f64, rand_vec_mf64
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


export depth


@inline function depth(network::ComparatorNetwork{N}) where {N}
    generation = ntuple(_ -> 0, Val{N}())
    @inbounds for (i, j) in network.comparators
        age = max(generation[i], generation[j]) + 1
        generation = Base.setindex(generation, age, i)
        generation = Base.setindex(generation, age, j)
    end
    return maximum(generation)
end


################################################################ CANONICAL FORMS


export canonize


struct Instruction{T}
    opcode::Symbol
    outputs::Vector{T}
    inputs::Vector{T}
end


function _comparator_code(network::ComparatorNetwork{N}) where {N}
    generation = [1 for _ = 1:N]
    code = Instruction{Tuple{Int,Int}}[]
    for (x, y) in network.comparators
        x = Int(x)
        y = Int(y)
        @assert x < y
        gen_x = generation[x]
        gen_y = generation[y]
        outputs = [(x, gen_x + 1), (y, gen_y + 1)]
        inputs = [(x, gen_x), (y, gen_y)]
        push!(code, Instruction(:comparator, outputs, inputs))
        generation[x] = gen_x + 1
        generation[y] = gen_y + 1
    end
    outputs = [(i, generation[i]) for i = 1:N]
    inputs = [(i, 1) for i = 1:N]
    return (code, outputs, inputs)
end


function _is_valid_ssa(
    code::AbstractVector{Instruction{T}},
    outputs::AbstractVector{T},
    inputs::AbstractVector{T},
) where {T}
    computed = Set{T}(inputs)
    for instr in code
        for input in instr.inputs
            if !(input in computed)
                return false
            end
        end
        for output in instr.outputs
            if output in computed
                return false
            end
            push!(computed, output)
        end
    end
    return issubset(outputs, computed)
end


function _eliminate_dead_code!(
    code::AbstractVector{Instruction{T}},
    outputs::AbstractVector{T},
) where {T}
    needed = Set{T}(outputs)
    dead_indices = BitSet()
    for (index, instr) in Iterators.reverse(pairs(code))
        if any(output in needed for output in instr.outputs)
            for input in instr.inputs
                push!(needed, input)
            end
        else
            push!(dead_indices, index)
        end
    end
    deleteat!(code, dead_indices)
    return !isempty(dead_indices)
end


function _delete_duplicate_comparators!(
    code::AbstractVector{Instruction{T}},
    outputs::AbstractVector{T},
) where {T}
    compared = Dict{T,T}()
    identical = Dict{T,T}()
    dead_indices = BitSet()
    for (index, instr) in pairs(code)
        @assert instr.opcode == :comparator
        @assert length(instr.outputs) == 2
        @assert length(instr.inputs) == 2
        x_out, y_out = instr.outputs
        x_in, y_in = instr.inputs
        if ((haskey(compared, x_in) && (compared[x_in] == y_in)) ||
            (haskey(compared, y_in) && (compared[y_in] == x_in)))
            @assert (compared[x_in] == y_in) && (compared[y_in] == x_in)
            push!(dead_indices, index)
            identical[x_out] = x_in
            identical[y_out] = y_in
        else
            compared[x_out] = y_out
            compared[y_out] = x_out
        end
    end
    deleteat!(code, dead_indices)
    for instr in code
        for (index, input) in pairs(instr.inputs)
            if haskey(identical, input)
                instr.inputs[index] = identical[input]
            end
        end
        for (index, output) in pairs(instr.outputs)
            if haskey(identical, output)
                instr.outputs[index] = identical[output]
            end
        end
    end
    for (index, output) in pairs(outputs)
        if haskey(identical, output)
            outputs[index] = identical[output]
        end
    end
    return !isempty(dead_indices)
end


function _canonize_code!(
    code::AbstractVector{Instruction{T}},
    outputs::AbstractVector{T},
    inputs::AbstractVector{T},
) where {T}
    while true
        changed = false
        changed |= _eliminate_dead_code!(code, outputs)
        @assert _is_valid_ssa(code, outputs, inputs)
        changed |= _delete_duplicate_comparators!(code, outputs)
        @assert _is_valid_ssa(code, outputs, inputs)
        if !changed
            break
        end
    end
    if isempty(code)
        # Early return necessary to avoid calling `reduce(vcat, [])`.
        return code
    end
    generation = Dict{T,Int}()
    blocks = Vector{Vector{Instruction{T}}}()
    for input in inputs
        generation[input] = 0
    end
    for instr in code
        gen = 0
        for input in instr.inputs
            @assert haskey(generation, input)
            gen = max(gen, generation[input])
        end
        gen += 1
        if gen <= length(blocks)
            push!(blocks[gen], instr)
        else
            @assert gen == length(blocks) + 1
            push!(blocks, [instr])
        end
        for output in instr.outputs
            @assert !haskey(generation, output)
            generation[output] = gen
        end
    end
    for block in blocks
        sort!(block, by=instr -> instr.outputs)
    end
    return reduce(vcat, blocks)
end


function canonize(network::ComparatorNetwork{N}) where {N}
    code, outputs, inputs = _comparator_code(network)
    return ComparatorNetwork{N}([
        (UInt8(instr.outputs[1][1]), UInt8(instr.outputs[2][1]))
        for instr in _canonize_code!(code, outputs, inputs)])
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
    Base.require_one_based_indexing(data)
    @assert length(data) == N
    return _unsafe_run_comparator_network!(data, network, comparator)
end


function run_comparator_network(
    input::NTuple{N,T},
    network::ComparatorNetwork{N},
    comparator::C,
) where {N,T,C}
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


################################################## COMPARATOR NETWORK GENERATION


export test_comparator_network, prune!, generate_comparator_network


@inline function _unsafe_test_comparator_network!(
    temp::AbstractVector{T},
    network::ComparatorNetwork{N},
    comparator::C,
    test_cases::AbstractVector{NTuple{N,T}},
    property::P,
) where {N,T,C,P}
    # Assumes: temp has indices 1:N.
    # Assumes: network is well-formed (comparator indices lie in 1:N).
    for test_case in test_cases
        @simd ivdep for i = 1:N
            @inbounds temp[i] = test_case[i]
        end
        _unsafe_run_comparator_network!(temp, network, comparator)
        @inline correct = property(temp)
        if !correct
            return false
        end
    end
    return true
end


test_comparator_network(
    network::ComparatorNetwork{N},
    comparator::C,
    test_cases::AbstractVector{NTuple{N,T}},
    property::P,
) where {N,T,C,P} = _unsafe_test_comparator_network!(
    Vector{T}(undef, N), network, comparator, test_cases, property)


function _unsafe_prune!(
    temp::AbstractVector{T},
    network::ComparatorNetwork{N},
    comparator::C,
    test_cases::AbstractVector{NTuple{N,T}},
    property::P,
) where {N,T,C,P}
    while true
        found = false
        for i in shuffle(1:length(network.comparators))
            original_comparator = network.comparators[i]
            deleteat!(network.comparators, i)
            pass = _unsafe_test_comparator_network!(
                temp, network, comparator, test_cases, property)
            if pass
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


prune!(
    network::ComparatorNetwork{N},
    comparator::C,
    test_cases::AbstractVector{NTuple{N,T}},
    property::P,
) where {N,T,C,P} = _unsafe_prune!(
    Vector{T}(undef, N), network, comparator, test_cases, property)


@inline function _random_comparator(::Val{N}) where {N}
    i = rand(0x01:UInt8(N))
    j = rand(0x01:UInt8(N - 1))
    j += UInt8(j >= i)
    return minmax(i, j)
end


function _unsafe_generate_comparator_network!(
    temp::AbstractVector{T},
    comparator::C,
    test_cases::AbstractVector{NTuple{N,T}},
    property::P,
) where {N,T,C,P}
    network = ComparatorNetwork{N}(Tuple{UInt8,UInt8}[])
    while true
        pass = _unsafe_test_comparator_network!(
            temp, network, comparator, test_cases, property)
        if pass
            break
        else
            push!(network.comparators, _random_comparator(Val{N}()))
        end
    end
    return _unsafe_prune!(temp, network, comparator, test_cases, property)
end


generate_comparator_network(
    comparator::C,
    test_cases::AbstractVector{NTuple{N,T}},
    property::P,
) where {N,T,C,P} = _unsafe_generate_comparator_network!(
    Vector{T}(undef, N), comparator, test_cases, property)


include("Annealing.jl")
using .Annealing
export pareto_dominates, pareto_push!, optimize_comparator_network


######################################################### TEST CASE OPTIMIZATION


export mfadd_relative_error, optimize_mfadd_relative_error,
    find_worst_case_mfadd_inputs


@inline function _unsafe_mfadd_relative_error!(
    temp::AbstractVector{T},
    network::ComparatorNetwork{N},
    x::NTuple{X,T},
    y::NTuple{Y,T},
    ::Val{Z},
) where {N,X,Y,Z,T}
    @assert X + Y == N
    @assert 0 < Z <= N
    data = riffle(x, y)
    @simd ivdep for i = 1:N
        @inbounds temp[i] = data[i]
    end
    _unsafe_run_comparator_network!(temp, network, two_sum)
    head = ntuple(i -> (@inbounds temp[i]), Val{Z}())
    normalized_head = alternating_normalize(head)
    if head !== normalized_head
        return one(T)
    end
    if Z == N
        return zero(T)
    end
    tail = ntuple(i -> (@inbounds temp[Z+i]), Val{N - Z}())
    normalized_tail = alternating_normalize(tail)
    first_kept = abs(first(normalized_head))
    first_discarded = abs(first(normalized_tail))
    if iszero(first_kept) & iszero(first_discarded)
        return zero(T)
    elseif first_discarded < first_kept
        return first_discarded / first_kept
    else
        return one(T)
    end
end


mfadd_relative_error(
    network::ComparatorNetwork{N},
    x::NTuple{X,T},
    y::NTuple{Y,T},
    ::Val{Z},
) where {N,X,Y,Z,T} = _unsafe_mfadd_relative_error!(
    Vector{T}(undef, N), network, x, y, Val{Z}())


@inline function _flip_bit(x::T, n::Int) where {T}
    u = reinterpret(Unsigned, x)
    return reinterpret(T, xor(u, one(typeof(u)) << n))
end


const BITS_PER_BYTE = 8
@assert BITS_PER_BYTE * sizeof(UInt32) == 32
@assert BITS_PER_BYTE * sizeof(UInt64) == 64


@inline function _flip_bit(x::NTuple{N,T}, n::Int) where {N,T}
    bit_size = BITS_PER_BYTE * sizeof(T)
    item_index, bit_index = divrem(n, bit_size)
    item_index += 1
    return @inbounds Base.setindex(
        x, _flip_bit(x[item_index], bit_index), item_index)
end


@inline function _unsafe_optimize_mfadd_relative_error!(
    temp::AbstractVector{T},
    network::ComparatorNetwork{N},
    x::NTuple{X,T},
    y::NTuple{Y,T},
    ::Val{Z},
) where {N,X,Y,Z,T}
    max_error = _unsafe_mfadd_relative_error!(
        temp, network, x, y, Val{Z}())
    while true
        new_x = x
        new_y = y
        for i = 1:X*BITS_PER_BYTE*sizeof(T)
            flip_x = alternating_normalize(_flip_bit(x, i - 1))
            if all(isfinite, flip_x)
                flip_error = _unsafe_mfadd_relative_error!(
                    temp, network, flip_x, new_y, Val{Z}())
                if flip_error > max_error
                    new_x = flip_x
                    max_error = flip_error
                end
            end
        end
        for i = 1:Y*BITS_PER_BYTE*sizeof(T)
            flip_y = alternating_normalize(_flip_bit(y, i - 1))
            if all(isfinite, flip_y)
                flip_error = _unsafe_mfadd_relative_error!(
                    temp, network, new_x, flip_y, Val{Z}())
                if flip_error > max_error
                    new_y = flip_y
                    max_error = flip_error
                end
            end
        end
        if (new_x === x) & (new_y === y)
            return (max_error, x, y)
        else
            x = new_x
            y = new_y
        end
    end
end


optimize_mfadd_relative_error(
    network::ComparatorNetwork{N},
    x::NTuple{X,T},
    y::NTuple{Y,T},
    ::Val{Z},
) where {N,X,Y,Z,T} = _unsafe_optimize_mfadd_relative_error!(
    Vector{T}(undef, N), network, x, y, Val{Z}())


@inline _vec_slice(data::NTuple{N,Vec{M,T}}, i::Int) where {M,N,T} =
    ntuple(j -> (@inbounds data[j][i]), Val{N}())


function find_worst_case_mfadd_inputs(
    network::ComparatorNetwork{N},
    ::Val{X},
    ::Val{Y},
    ::Val{Z},
    ::Val{M},
    duration_ns::UInt64,
) where {N,X,Y,Z,M}
    start_ns = time_ns()
    @assert X + Y == N
    @assert 0 < Z <= N
    max_error = zero(Float64)
    worst_case_x = ntuple(_ -> zero(Float64), Val{X}())
    worst_case_y = ntuple(_ -> zero(Float64), Val{Y}())
    count = 0
    temp_vector = Vector{Vec{M,Float64}}(undef, N)
    temp_scalar = Vector{Float64}(undef, N)
    while true
        x = rand_vec_mf64(Val{M}(), Val{X}())
        y = rand_vec_mf64(Val{M}(), Val{Y}())
        data = riffle(x, y)
        @simd ivdep for i = 1:N
            @inbounds temp_vector[i] = data[i]
        end
        _unsafe_run_comparator_network!(temp_vector, network, two_sum)
        head = ntuple(i -> (@inbounds temp_vector[i]), Val{Z}())
        normalized_head = alternating_normalize(head)
        if head !== normalized_head
            @inbounds for i = 1:M
                head_slice = _vec_slice(head, i)
                normalized_slice = _vec_slice(normalized_head, i)
                count += 1
                if head_slice !== normalized_slice
                    return (one(Float64), count,
                        _vec_slice(x, i), _vec_slice(y, i))
                end
            end
            @assert false
        end
        if Z < N
            tail = ntuple(i -> (@inbounds temp_vector[Z+i]), Val{N - Z}())
            normalized_tail = alternating_normalize(tail)
            first_kept = abs(first(normalized_head))
            first_discarded = abs(first(normalized_tail))
            if !all(first_discarded < first_kept)
                @inbounds for i = 1:M
                    count += 1
                    if !(first_discarded[i] < first_kept[i])
                        return (one(Float64), count,
                            _vec_slice(x, i), _vec_slice(y, i))
                    end
                end
                @assert false
            end
            relative_error = first_discarded / first_kept
            @inbounds for i = 1:M
                if relative_error[i] > max_error
                    max_error, worst_case_x, worst_case_y =
                        _unsafe_optimize_mfadd_relative_error!(
                            temp_scalar, network,
                            _vec_slice(x, i), _vec_slice(y, i), Val{Z}())
                end
            end
        end
        count += M
        if time_ns() - start_ns >= duration_ns
            return (max_error, count, worst_case_x, worst_case_y)
        end
    end
end


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
