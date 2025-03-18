module TestCaseGenerators

using SIMD: Vec

using ..Comparators: alternating_normalize

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


@inline _rand_vec_u64(::Val{N}) where {N} =
    Vec{N,UInt64}(ntuple(_ -> rand(UInt64), Val{N}()))


@inline function _rand_vec_u16(::Val{N}) where {N}
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
    sign_exponent_data = _rand_vec_u16(Val{N}())
    sign_bits = (sign_exponent_data << 48) & 0x8000000000000000
    exponents = ((sign_exponent_data & 0x03FF) + 0x0200) << 52
    mantissa_data = _rand_vec_u64(Val{N}())
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


################################################################################

end # module TestCaseGenerators
