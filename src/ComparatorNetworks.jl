module ComparatorNetworks

using Printf: @sprintf
using Random: shuffle
using SIMD: Vec

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


########################################################### COMPARATOR FUNCTIONS


export bitminmax, two_sum


@inline bitminmax(x, y) = (x & y, x | y)


@inline function two_sum(x, y)
    s = x + y
    x_prime = s - y
    y_prime = s - x_prime
    delta_x = x - x_prime
    delta_y = y - y_prime
    e = delta_x + delta_y
    return (s, e)
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


######################################################### CORRECTNESS PROPERTIES


export bitsorted


@inline function bitsorted(data::AbstractVector{T}) where {T}
    result = ~zero(T)
    @simd for i = firstindex(data):lastindex(data)-1
        @inbounds result &= data[i+1] | ~data[i]
    end
    return all(iszero(~result))
end


################################################## COMPARATOR NETWORK GENERATION


export test_comparator_network, prune!, generate_comparator_network,
    mutate_comparator_network!


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


function _unsafe_mutate_comparator_network!(
    temp::AbstractVector{T},
    network::ComparatorNetwork{N},
    comparator::C,
    test_cases::AbstractVector{NTuple{N,T}},
    property::P,
    insert_weight::Int,
    delete_weight::Int,
    swap_weight::Int,
    duration_ns::UInt64,
) where {N,T,C,P}
    num_comparators = length(network.comparators)
    w1 = insert_weight
    w2 = w1 + delete_weight
    w3 = w2 + swap_weight
    start_ns = time_ns()
    while true
        w = rand(1:w3)
        if w <= w1 # insert
            i = rand(1:num_comparators+1)
            insert!(network.comparators, i, _random_comparator(Val{N}()))
            pass = _unsafe_test_comparator_network!(
                temp, network, comparator, test_cases, property)
            if pass
                return network
            end
            deleteat!(network.comparators, i)
        elseif (w <= w2) & (num_comparators >= 1) # delete
            i = rand(1:num_comparators)
            @inbounds original_comparator = network.comparators[i]
            deleteat!(network.comparators, i)
            pass = _unsafe_test_comparator_network!(
                temp, network, comparator, test_cases, property)
            if pass
                return network
            end
            insert!(network.comparators, i, original_comparator)
        elseif (w <= w3) & (num_comparators >= 2) # swap
            i = rand(1:num_comparators)
            j = rand(1:num_comparators-1)
            j += (j >= i)
            @inbounds network.comparators[i], network.comparators[j] =
                network.comparators[j], network.comparators[i]
            pass = _unsafe_test_comparator_network!(
                temp, network, comparator, test_cases, property)
            if pass
                return network
            end
            @inbounds network.comparators[i], network.comparators[j] =
                network.comparators[j], network.comparators[i]
        end
        if time_ns() - start_ns >= duration_ns
            return network
        end
    end
end


mutate_comparator_network!(
    network::ComparatorNetwork{N},
    comparator::C,
    test_cases::AbstractVector{NTuple{N,T}},
    property::P,
    insert_weight::Int,
    delete_weight::Int,
    swap_weight::Int,
    duration_ns::UInt64,
) where {N,T,C,P} = _unsafe_mutate_comparator_network!(
    Vector{T}(undef, N), network, comparator, test_cases, property,
    insert_weight, delete_weight, swap_weight, duration_ns)


############################################################## COMPARATOR PASSES


export top_down_bitbubble, top_down_accumulate, top_down_bitsort,
    top_down_normalize, bottom_up_bitbubble, bottom_up_accumulate,
    bottom_up_bitsort, bottom_up_normalize, alternating_bitbubble,
    alternating_accumulate, alternating_bitsort, alternating_normalize


function _inline_pass_expr(comparator::Symbol, method::Symbol, N::Int)
    xs = [Symbol('x', i) for i in Base.OneTo(N)]
    body = Expr[]
    push!(body, Expr(:meta, :inline))
    push!(body, Expr(:(=), Expr(:tuple, xs...), :x))
    if method == :top_down
        for i in 1:N-1
            push!(body, Expr(:(=), Expr(:tuple, xs[i], xs[i+1]),
                Expr(:call, comparator, xs[i], xs[i+1])))
        end
    elseif method == :bottom_up
        for i = N-1:-1:1
            push!(body, Expr(:(=), Expr(:tuple, xs[i], xs[i+1]),
                Expr(:call, comparator, xs[i], xs[i+1])))
        end
    elseif method == :riffle
        for i = 1:2:N-1
            push!(body, Expr(:(=), Expr(:tuple, xs[i], xs[i+1]),
                Expr(:call, comparator, xs[i], xs[i+1])))
        end
        for i = 2:2:N-1
            push!(body, Expr(:(=), Expr(:tuple, xs[i], xs[i+1]),
                Expr(:call, comparator, xs[i], xs[i+1])))
        end
    else
        @assert false
    end
    push!(body, Expr(:return, Expr(:tuple, xs...)))
    return Expr(:block, body...)
end


@generated function top_down_bitbubble(x::NTuple{N,T}) where {N,T}
    return _inline_pass_expr(:bitminmax, :top_down, N)
end


@generated function top_down_accumulate(x::NTuple{N,T}) where {N,T}
    return _inline_pass_expr(:two_sum, :top_down, N)
end


@inline function top_down_bitsort(x::NTuple{N,T}) where {N,T}
    while true
        x_next = top_down_bitbubble(x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


@inline function top_down_normalize(x::NTuple{N,T}) where {N,T}
    while true
        x_next = top_down_accumulate(x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


@generated function bottom_up_bitbubble(x::NTuple{N,T}) where {N,T}
    return _inline_pass_expr(:bitminmax, :bottom_up, N)
end


@generated function bottom_up_accumulate(x::NTuple{N,T}) where {N,T}
    return _inline_pass_expr(:two_sum, :bottom_up, N)
end


@inline function bottom_up_bitsort(x::NTuple{N,T}) where {N,T}
    while true
        x_next = bottom_up_bitbubble(x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


@inline function bottom_up_normalize(x::NTuple{N,T}) where {N,T}
    while true
        x_next = bottom_up_accumulate(x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


@generated function alternating_bitbubble(x::NTuple{N,T}) where {N,T}
    return _inline_pass_expr(:bitminmax, :riffle, N)
end


@generated function alternating_accumulate(x::NTuple{N,T}) where {N,T}
    return _inline_pass_expr(:two_sum, :riffle, N)
end


@inline function alternating_bitsort(x::NTuple{N,T}) where {N,T}
    while true
        x_next = alternating_bitbubble(x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


@inline function alternating_normalize(x::NTuple{N,T}) where {N,T}
    while true
        x_next = alternating_accumulate(x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


#################################################### BINARY TEST CASE GENERATION


export all_bit_vectors


@inline all_bit_vectors(::Val{0}) = ()


@inline all_bit_vectors(::Val{1}) = (0b01,)


@inline all_bit_vectors(::Val{2}) = (0b0011, 0b0101)


@inline all_bit_vectors(::Val{3}) = (0x0F, 0x33, 0x55)


@inline all_bit_vectors(::Val{4}) = (0x00FF, 0x0F0F, 0x3333, 0x5555)


@inline all_bit_vectors(::Val{5}) =
    (0x00000001, 0x00010117, 0x0117177F, 0x177F7FFF, 0x7FFFFFFF)


@inline all_bit_vectors(::Val{6}) = (
    0x00000000FFFFFFFF, 0x0000FFFF0000FFFF, 0x00FF00FF00FF00FF,
    0x0F0F0F0F0F0F0F0F, 0x3333333333333333, 0x5555555555555555)


@inline all_bit_vectors(::Val{7}) = (
    Vec{2,UInt64}((0x0000000000000000, 0xFFFFFFFFFFFFFFFF)),
    Vec{2,UInt64}((0x00000000FFFFFFFF, 0x00000000FFFFFFFF)),
    Vec{2,UInt64}((0x0000FFFF0000FFFF, 0x0000FFFF0000FFFF)),
    Vec{2,UInt64}((0x00FF00FF00FF00FF, 0x00FF00FF00FF00FF)),
    Vec{2,UInt64}((0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F)),
    Vec{2,UInt64}((0x3333333333333333, 0x3333333333333333)),
    Vec{2,UInt64}((0x5555555555555555, 0x5555555555555555)))


@inline _vec_concat(v::Vec{M,T}, w::Vec{N,T}) where {M,N,T} =
    Vec{M + N,T}((v.data..., w.data...))


@inline function all_bit_vectors(::Val{N}) where {N}
    _zero = zero(Vec{1 << (N - 7),UInt64})
    _one = ~_zero
    first_row = _vec_concat(_zero, _one)
    prev = all_bit_vectors(Val{N - 1}())
    remaining_rows = _vec_concat.(prev, prev)
    return (first_row, remaining_rows...)
end


############################################ FLOATING-POINT TEST CASE GENERATION


@inline _rand_vec_64(::Val{N}) where {N} =
    Vec{N,UInt64}(ntuple(_ -> rand(UInt64), Val{N}()))


@inline function _rand_vec_16(::Val{N}) where {N}
    data = ntuple(_ -> rand(UInt64), Val{(N + 3) >> 2}())
    return Vec{N,UInt64}(ntuple(
        i -> (data[(i+3)>>2] >> ((i & 3) << 4)) & 0xFFFF, Val{N}()))
end


# Reduce an integer from the range [0, 63] to the range [0, 52].
@inline _reduce_63_to_52(i) = ((i << 5) + (i << 4) + (i << 2) + 0x0038) >> 6


# Copy bit 0 (LSB) of x to bit positions 0 through n-1.
@inline function _copy_low_bits(x, n)
    bit = 0x0000000000000001
    shifted_bit = (bit << n)
    low_bit = x & bit
    mask = shifted_bit - bit
    value = shifted_bit - low_bit
    return (x & ~mask) | (value & mask)
end


# Copy bit 51 of x to bit positions 51 through 51-(n-1).
@inline function _copy_high_bits(x, n)
    high_bit = (x & 0x0008000000000000) >> 51
    mask = (0x000FFFFFFFFFFFFF >> n) << n
    value = 0x0010000000000000 - high_bit
    return (x & ~mask) | (value & mask)
end


# Generate a random vector of 64-bit floating-point numbers for testing
# floating-point accumulation networks (FPANs). The generated values have
# uniformly distributed signs and exponents in the range [-511, +512], and
# their mantissas are biased to have many leading/trailing zeros/ones.
@inline function _rand_vec_f64(::Val{N}) where {N}
    sign_exponent_data = _rand_vec_16(Val{N}())
    sign_bits = (sign_exponent_data << 48) & 0x8000000000000000
    exponents = ((sign_exponent_data & 0x03FF) + 0x0200) << 52
    mantissa_data = _rand_vec_64(Val{N}())
    i = mantissa_data >> 58
    j = (mantissa_data >> 52) & 0x3F
    low_index = _reduce_63_to_52(min(i, j))
    high_index = _reduce_63_to_52(max(i, j))
    mantissas = mantissa_data & 0x000FFFFFFFFFFFFF
    mantissas = _copy_low_bits(mantissas, low_index)
    mantissas = _copy_high_bits(mantissas, high_index)
    return reinterpret(Vec{N,Float64}, sign_bits | exponents | mantissas)
end


@inline _rand_vec_mf64(::Val{M}, ::Val{N}) where {M,N} =
    alternating_normalize(ntuple(_ -> _rand_vec_f64(Val{M}()), Val{N}()))


######################################################### TEST CASE MANIPULATION


export riffle!, riffle


@inline function riffle!(
    v::AbstractVector{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
) where {T}
    # Validate array dimensions.
    Base.require_one_based_indexing(v, x, y)
    len_x = length(x)
    len_y = length(y)
    @assert length(v) == len_x + len_y
    # Riffle common elements.
    @simd ivdep for i = 1:min(len_x, len_y)
        _2i = i + i
        @inbounds v[_2i-1] = x[i]
        @inbounds v[_2i] = y[i]
    end
    # Append remaining elements.
    if len_x > len_y
        copyto!(v, len_y + len_y + 1, x, len_y + 1, len_x - len_y)
    elseif len_x < len_y
        copyto!(v, len_x + len_x + 1, y, len_x + 1, len_y - len_x)
    end
    return v
end


@generated function riffle(x::NTuple{M,T}, y::NTuple{N,T}) where {M,N,T}
    xs = [Symbol('x', i) for i in Base.OneTo(M)]
    ys = [Symbol('y', i) for i in Base.OneTo(N)]
    vs = riffle!(Vector{Symbol}(undef, M + N), xs, ys)
    return Expr(:block, Expr(:meta, :inline),
        Expr(:(=), Expr(:tuple, xs...), :x),
        Expr(:(=), Expr(:tuple, ys...), :y),
        Expr(:return, Expr(:tuple, vs...)))
end


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
    @assert X + Y == N
    @assert 0 < Z <= N
    start_ns = time_ns()
    max_error = zero(Float64)
    worst_case_x = ntuple(_ -> zero(Float64), Val{X}())
    worst_case_y = ntuple(_ -> zero(Float64), Val{Y}())
    count = 0
    temp_vector = Vector{Vec{M,Float64}}(undef, N)
    temp_scalar = Vector{Float64}(undef, N)
    while true
        x = _rand_vec_mf64(Val{M}(), Val{X}())
        y = _rand_vec_mf64(Val{M}(), Val{Y}())
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


############################################################### GRAPHICAL OUTPUT


_svg_string(x::Float64) = rstrip(rstrip(@sprintf("%.15f", x), '0'), '.')


function _svg_string(
    x_min::Float64, x_max::Float64,
    y_min::Float64, y_max::Float64,
)
    width_string = _svg_string(x_max - x_min)
    height_string = _svg_string(y_max - y_min)
    return @sprintf(
        """<svg xmlns="%s" viewBox="%s %s %s %s" width="%s" height="%s">""",
        "http://www.w3.org/2000/svg", _svg_string(x_min), _svg_string(y_min),
        width_string, height_string, width_string, height_string)
end


struct _SVGCircle
    cx::Float64
    cy::Float64
    r::Float64
end


@inline _min_x(circle::_SVGCircle) = circle.cx - circle.r
@inline _min_y(circle::_SVGCircle) = circle.cy - circle.r
@inline _max_x(circle::_SVGCircle) = circle.cx + circle.r
@inline _max_y(circle::_SVGCircle) = circle.cy + circle.r


_svg_string(circle::_SVGCircle, color::String, width::Float64) = @sprintf(
    """<circle cx="%s" cy="%s" r="%s" stroke="%s" stroke-width="%s" fill="none" />""",
    _svg_string(circle.cx), _svg_string(circle.cy), _svg_string(circle.r),
    color, _svg_string(width))


struct _SVGLine
    x1::Float64
    y1::Float64
    x2::Float64
    y2::Float64
end


@inline _min_x(line::_SVGLine) = min(line.x1, line.x2)
@inline _min_y(line::_SVGLine) = min(line.y1, line.y2)
@inline _max_x(line::_SVGLine) = max(line.x1, line.x2)
@inline _max_y(line::_SVGLine) = max(line.y1, line.y2)


_svg_string(line::_SVGLine, color::String, width::Float64) = @sprintf(
    """<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s" />""",
    _svg_string(line.x1), _svg_string(line.y1),
    _svg_string(line.x2), _svg_string(line.y2),
    color, _svg_string(width))


@enum _ColumnOccupation begin
    _NONE
    _WEAK
    _STRONG
end


function _svg_string(
    network::ComparatorNetwork{N};
    line_height::Float64=32.0,
    circle_radius::Float64=8.0,
    minor_width::Float64=NaN,
    major_width::Float64=NaN,
    padding_left::Float64=NaN,
    padding_right::Float64=NaN,
    padding_top::Float64=NaN,
    padding_bottom::Float64=NaN,
) where {N}

    # `line_height` and `circle_radius` must be supplied by the user.
    @assert isfinite(line_height)
    @assert isfinite(circle_radius)

    # The remaining parameters are optional with the following default values.
    isfinite(minor_width) || (minor_width = 2.0 * circle_radius)
    isfinite(major_width) || (major_width = 5.0 * circle_radius)
    isfinite(padding_left) || (padding_left = 3.0 * circle_radius)
    isfinite(padding_right) || (padding_right = 3.0 * circle_radius)
    isfinite(padding_top) || (padding_top = 2.0 * circle_radius)
    isfinite(padding_bottom) || (padding_bottom = 2.0 * circle_radius)

    lines = _SVGLine[]
    circles = _SVGCircle[]
    occupied = _ColumnOccupation[_NONE for _ = 1:N]
    x = 0.0

    for (i, j) in network.comparators
        @assert 1 <= i < j <= N
        if (occupied[i] == _STRONG) || (occupied[j] == _STRONG)
            occupied .= _NONE
            x += major_width
        elseif any(occupied[k] != _NONE for k = i:j)
            x += minor_width
        end
        for k = i:j
            if occupied[k] == _NONE
                occupied[k] = _WEAK
            end
        end
        occupied[i] = _STRONG
        occupied[j] = _STRONG
        yi = i * line_height
        yj = j * line_height
        push!(lines, _SVGLine(x, yi - circle_radius, x, yj + circle_radius))
        push!(circles, _SVGCircle(x, yi, circle_radius))
        push!(circles, _SVGCircle(x, yj, circle_radius))
    end

    for i = 1:N
        push!(lines, _SVGLine(
            -padding_left, i * line_height,
            x + padding_right, i * line_height))
    end

    buf = IOBuffer()
    print(buf, _svg_string(-padding_left, x + padding_right,
        line_height - padding_top, N * line_height + padding_bottom))

    for line in lines
        print(buf, _svg_string(line, "white", 3.0))
    end
    for line in lines
        print(buf, _svg_string(line, "black", 1.0))
    end
    for circle in circles
        print(buf, _svg_string(circle, "white", 3.0))
    end
    for circle in circles
        print(buf, _svg_string(circle, "black", 1.0))
    end

    print(buf, "</svg>")
    return String(take!(buf))
end


function Base.show(
    io::IO,
    ::MIME"text/html",
    network::ComparatorNetwork{N},
) where {N}
    println(io, _svg_string(network))
end


################################################################################

end # module ComparatorNetworks
