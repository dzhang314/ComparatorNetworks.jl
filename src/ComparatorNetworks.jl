module ComparatorNetworks

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
    @inline length(network.comparators)
@inline Base.:(==)(a::ComparatorNetwork{N}, b::ComparatorNetwork{N}) where {N} =
    @inline (a.comparators == b.comparators)
@inline Base.copy(network::ComparatorNetwork{N}) where {N} =
    @inline ComparatorNetwork{N}(network.comparators)
@inline Base.hash(network::ComparatorNetwork{N}, h::UInt) where {N} =
    @inline hash(network.comparators, hash(N, h))
@inline Base.isless(
    a::ComparatorNetwork{M},
    b::ComparatorNetwork{N},
) where {M,N} = @inline isless(
    (M, length(a.comparators), a.comparators),
    (N, length(b.comparators), b.comparators))


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


function _code_depth(
    code::AbstractVector{Instruction{T}},
    outputs::AbstractVector{T},
    inputs::AbstractVector{T},
) where {T}
    generation = Dict{T,Int}()
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
        for output in instr.outputs
            @assert !haskey(generation, output)
            generation[output] = gen
        end
    end
    result = 0
    for output in outputs
        @assert haskey(generation, output)
        result = max(result, generation[output])
    end
    return result
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
    comparator::F,
) where {T,N,F}
    for (i, j) in network.comparators
        @inbounds @inline data[i], data[j] = comparator(data[i], data[j])
    end
    return data
end


@inline function run_comparator_network!(
    data::AbstractVector{T},
    network::ComparatorNetwork{N},
    comparator::F,
) where {T,N,F}
    Base.require_one_based_indexing(data)
    @assert length(data) == N
    return _unsafe_run_comparator_network!(data, network, comparator)
end


@inline function run_comparator_network(
    input::NTuple{N,T},
    network::ComparatorNetwork{N},
    comparator::F,
) where {T,N,F}
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
        @assert i < j
        push!(body, Expr(:(=), Expr(:tuple, xs[i], xs[j]),
            Expr(:call, comparator, xs[i], xs[j])))
    end
    push!(body, Expr(:return, Expr(:tuple, xs...)))
    return Expr(:->, :x, Expr(:block, body...))
end


############################################################ CRITERION FUNCTIONS


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
    data::AbstractVector{T},
    network::ComparatorNetwork{N},
    comparator::F,
    test_cases::AbstractVector{NTuple{N,T}},
    criterion::G,
) where {N,T,F,G}
    for test_case in test_cases
        @simd ivdep for i = 1:N
            @inbounds data[i] = test_case[i]
        end
        _unsafe_run_comparator_network!(data, network, comparator)
        @inline correct = criterion(data)
        if !correct
            return false
        end
    end
    return true
end


@inline test_comparator_network(
    network::ComparatorNetwork{N},
    comparator::F,
    test_cases::AbstractVector{NTuple{N,T}},
    criterion::G,
) where {N,T,F,G} = _unsafe_test_comparator_network!(
    Vector{T}(undef, N), network, comparator, test_cases, criterion)


function _unsafe_prune!(
    data::AbstractVector{T},
    network::ComparatorNetwork{N},
    comparator::F,
    test_cases::AbstractVector{NTuple{N,T}},
    criterion::G,
) where {N,T,F,G}
    while true
        found = false
        for i in shuffle(1:length(network.comparators))
            original_comparator = network.comparators[i]
            deleteat!(network.comparators, i)
            pass = _unsafe_test_comparator_network!(
                data, network, comparator, test_cases, criterion)
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


@inline prune!(
    network::ComparatorNetwork{N},
    comparator::F,
    test_cases::AbstractVector{NTuple{N,T}},
    criterion::G,
) where {N,T,F,G} = _unsafe_prune!(
    Vector{T}(undef, N), network, comparator, test_cases, criterion)


@inline function _random_comparator(::Val{N}) where {N}
    i = rand(0x01:UInt8(N))
    j = rand(0x01:UInt8(N - 1))
    j += UInt8(j >= i)
    return minmax(i, j)
end


function _unsafe_generate_comparator_network!(
    data::AbstractVector{T},
    comparator::F,
    test_cases::AbstractVector{NTuple{N,T}},
    criterion::G,
) where {N,T,F,G}
    network = ComparatorNetwork{N}(Tuple{UInt8,UInt8}[])
    while true
        pass = _unsafe_test_comparator_network!(
            data, network, comparator, test_cases, criterion)
        if pass
            break
        else
            push!(network.comparators, _random_comparator(Val{N}()))
        end
    end
    return _unsafe_prune!(data, network, comparator, test_cases, criterion)
end


@inline generate_comparator_network(
    comparator::F,
    test_cases::AbstractVector{NTuple{N,T}},
    criterion::G,
) where {N,T,F,G} = _unsafe_generate_comparator_network!(
    Vector{T}(undef, N), comparator, test_cases, criterion)


function _unsafe_mutate_comparator_network!(
    data::AbstractVector{T},
    network::ComparatorNetwork{N},
    comparator::F,
    test_cases::AbstractVector{NTuple{N,T}},
    criterion::G,
    insert_weight::Int,
    delete_weight::Int,
    swap_weight::Int,
    duration_ns::UInt64,
) where {N,T,F,G}
    num_comparators = length(network.comparators)
    w1 = insert_weight
    w2 = w1 + delete_weight
    w3 = w2 + swap_weight
    start = time_ns()
    while true
        w = rand(1:w3)
        if w <= w1 # insert
            i = rand(1:num_comparators+1)
            insert!(network.comparators, i, _random_comparator(Val{N}()))
            pass = _unsafe_test_comparator_network!(
                data, network, comparator, test_cases, criterion)
            if pass
                return network
            end
            deleteat!(network.comparators, i)
        elseif w <= w2 # delete
            i = rand(1:num_comparators)
            @inbounds original_comparator = network.comparators[i]
            deleteat!(network.comparators, i)
            pass = _unsafe_test_comparator_network!(
                data, network, comparator, test_cases, criterion)
            if pass
                return network
            end
            insert!(network.comparators, i, original_comparator)
        else # swap
            i = rand(1:num_comparators)
            j = rand(1:num_comparators-1)
            j += (j >= i)
            @inbounds network.comparators[i], network.comparators[j] =
                network.comparators[j], network.comparators[i]
            pass = _unsafe_test_comparator_network!(
                data, network, comparator, test_cases, criterion)
            if pass
                return network
            end
            @inbounds network.comparators[i], network.comparators[j] =
                network.comparators[j], network.comparators[i]
        end
        if time_ns() - start >= duration_ns
            return network
        end
    end
end


@inline mutate_comparator_network!(
    network::ComparatorNetwork{N},
    comparator::F,
    test_cases::AbstractVector{NTuple{N,T}},
    criterion::G,
    insert_weight::Int,
    delete_weight::Int,
    swap_weight::Int,
    duration_ns::UInt64,
) where {N,T,F,G} = _unsafe_mutate_comparator_network!(
    Vector{T}(undef, N), network, comparator, test_cases, criterion,
    insert_weight, delete_weight, swap_weight, duration_ns)


############################################################## COMPARATOR PASSES


export top_down_bitbubble, top_down_accumulate, top_down_bitsort,
    top_down_normalize, bottom_up_bitbubble, bottom_up_accumulate,
    bottom_up_bitsort, bottom_up_normalize, riffle_bitbubble,
    riffle_accumulate, riffle_bitsort, riffle_normalize


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


@generated function riffle_bitbubble(x::NTuple{N,T}) where {N,T}
    return _inline_pass_expr(:bitminmax, :riffle, N)
end


@generated function riffle_accumulate(x::NTuple{N,T}) where {N,T}
    return _inline_pass_expr(:two_sum, :riffle, N)
end


@inline function riffle_bitsort(x::NTuple{N,T}) where {N,T}
    while true
        x_next = riffle_bitbubble(x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


@inline function riffle_normalize(x::NTuple{N,T}) where {N,T}
    while true
        x_next = riffle_accumulate(x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


#################################################### RANDOM TEST CASE GENERATION


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
    _riffle_normalize(ntuple(_ -> _rand_vec_f64(Val{M}()), Val{N}()))


######################################################### TEST CASE OPTIMIZATION


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
    return Base.setindex(x, _flip_bit(x[item_index], bit_index), item_index)
end


################################################################################

end # module ComparatorNetworks
