module ComparatorNetworks

using SIMD: Vec

################################################################################


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
    hash(network.comparators, h)


################################################################################


struct Instruction{T}
    opcode::Symbol
    outputs::Vector{T}
    inputs::Vector{T}
end


function sorting_code(network::ComparatorNetwork{N}) where {N}
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
        push!(code, Instruction(:minmax, outputs, inputs))
        generation[x] = gen_x + 1
        generation[y] = gen_y + 1
    end
    outputs = [(i, generation[i]) for i = 1:N]
    inputs = [(i, 1) for i = 1:N]
    return (code, outputs, inputs)
end


function accumulation_code(network::ComparatorNetwork{N}) where {N}
    generation = [1 for _ = 1:N]
    code = Instruction{Tuple{Int,Int}}[]
    k = N
    for (x, y) in network.comparators
        x = Int(x)
        y = Int(y)
        @assert x < y
        gen_x = generation[x]
        gen_y = generation[y]
        k += 1
        # TwoSum algorithm with temporary variables
        # k1 = x_prime, k2 = y_prime, k3 = delta_x, k4 = delta_y.
        push!(code, Instruction(:+, [(x, gen_x + 1)], [(x, gen_x), (y, gen_y)]))
        push!(code, Instruction(:-, [(k, 1)], [(x, gen_x + 1), (y, gen_y)]))
        push!(code, Instruction(:-, [(k, 2)], [(x, gen_x + 1), (k, 1)]))
        push!(code, Instruction(:-, [(k, 3)], [(x, gen_x), (k, 1)]))
        push!(code, Instruction(:-, [(k, 4)], [(y, gen_y), (k, 2)]))
        push!(code, Instruction(:+, [(y, gen_y + 1)], [(k, 3), (k, 4)]))
        generation[x] = gen_x + 1
        generation[y] = gen_y + 1
    end
    outputs = [(i, generation[i]) for i = 1:N]
    inputs = [(i, 1) for i = 1:N]
    return (code, outputs, inputs)
end


################################################################################


@inline _rand_vec_64() = Vec{8,UInt64}((
    rand(UInt64), rand(UInt64), rand(UInt64), rand(UInt64),
    rand(UInt64), rand(UInt64), rand(UInt64), rand(UInt64)))


@inline function _rand_vec_16()
    a = rand(UInt64)
    b = rand(UInt64)
    return Vec{8,UInt64}((
        a & 0xFFFF, (a >> 16) & 0xFFFF, (a >> 32) & 0xFFFF, a >> 48,
        b & 0xFFFF, (b >> 16) & 0xFFFF, (b >> 32) & 0xFFFF, b >> 48))
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
function _rand_vec_f64()
    sign_exponent_data = _rand_vec_16()
    sign_bits = (sign_exponent_data << 48) & 0x8000000000000000
    exponents = ((sign_exponent_data & 0x03FF) + 0x0200) << 52
    mantissa_data = _rand_vec_64()
    i = mantissa_data >> 58
    j = (mantissa_data >> 52) & 0x3F
    low_index = _reduce_63_to_52(min(i, j))
    high_index = _reduce_63_to_52(max(i, j))
    mantissas = mantissa_data & 0x000FFFFFFFFFFFFF
    mantissas = _copy_low_bits(mantissas, low_index)
    mantissas = _copy_high_bits(mantissas, high_index)
    return reinterpret(Vec{8,Float64}, sign_bits | exponents | mantissas)
end


################################################################################


@inline _bitwise_minmax(x, y) = (x & y, x | y)


@inline function _two_sum(x, y)
    s = x + y
    x_prime = s - y
    y_prime = s - x_prime
    delta_x = x - x_prime
    delta_y = y - y_prime
    e = delta_x + delta_y
    return (s, e)
end


function _generate_code(network::ComparatorNetwork{N}, comparator) where {N}
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


################################################################################

end # module ComparatorNetworks
