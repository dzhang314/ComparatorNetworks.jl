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
    @assert N > 7
    _zero = zero(Vec{1 << (N - 7),UInt64})
    _one = ~_zero
    first_row = _vec_concat(_zero, _one)
    prev = all_bit_vectors(Val{N - 1}())
    remaining_rows = _vec_concat.(prev, prev)
    return (first_row, remaining_rows...)
end


############################################# COMBINATORIAL TEST CASE GENERATION


export coarse_mfadd_test_cases


struct SubsetIterator
    n::Int
    k::Int
end


@inline Base.eltype(::SubsetIterator) = Vector{Int}
@inline Base.length(iter::SubsetIterator) =
    (iter.n >= iter.k >= 0) ? binomial(iter.n, iter.k) : 0


function Base.iterate(iter::SubsetIterator)
    if !(iter.n >= iter.k >= 0)
        return nothing
    end
    indices = collect(1:iter.k)
    return (indices, indices)
end


@inline function Base.iterate(iter::SubsetIterator, indices::Vector{Int})
    n, k = iter.n, iter.k
    @inbounds begin
        i = k
        while (i > 0) && (indices[i] == n - (k - i))
            i -= 1
        end
        if i == 0
            return nothing
        end
        origin = indices[i] + 1
        @simd ivdep for j in 0:k-i
            indices[i+j] = origin + j
        end
        return (indices, indices)
    end
end


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


@generated function riffle(x::NTuple{X,T}, y::NTuple{Y,T}) where {X,Y,T}
    xs = [Symbol('x', i) for i in Base.OneTo(X)]
    ys = [Symbol('y', i) for i in Base.OneTo(Y)]
    vs = riffle!(Vector{Symbol}(undef, X + Y), xs, ys)
    return Expr(:block, Expr(:meta, :inline),
        Expr(:(=), Expr(:tuple, xs...), :x),
        Expr(:(=), Expr(:tuple, ys...), :y),
        Expr(:return, Expr(:tuple, vs...)))
end


function coarse_mfadd_test_data(::Val{X}, ::Val{Y}) where {X,Y}
    result = NTuple{X + Y,UInt8}[]
    for s = 0:X
        x_padding = [0 for _ = 1:X-s]
        for t = 0:Y
            y_padding = [0 for _ = 1:Y-t]
            for d = min(s, t):-1:0
                n = s + t - d
                for duplicated in SubsetIterator(n, d)
                    remaining = setdiff(1:n, duplicated)
                    for x_indices in SubsetIterator(n - d, s - d)
                        y_indices = setdiff(1:n-d, x_indices)
                        x_data = UInt8.(Tuple(sort!(
                            vcat(x_padding, duplicated, remaining[x_indices]);
                            rev=true)))
                        y_data = UInt8.(Tuple(sort!(
                            vcat(y_padding, duplicated, remaining[y_indices]);
                            rev=true)))
                        push!(result, riffle(x_data, y_data))
                    end
                end
            end
        end
    end
    return result
end


function coarse_mfadd_test_cases(::Val{M}, ::Val{X}, ::Val{Y}) where {M,X,Y}
    data = coarse_mfadd_test_data(Val{X}(), Val{Y}())
    return [
        ntuple(
            j -> Vec(ntuple(
                i -> (i + M * (k - 1) <= length(data)) ?
                     data[i+M*(k-1)][j] : zero(UInt8),
                Val{M}())),
            Val{X + Y}())
        for k = 1:cld(length(data), M)]
end


############################################ FLOATING-POINT TEST CASE GENERATION


export rand_vec_f64, rand_vec_mf64


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
@inline function rand_vec_f64(::Val{N}) where {N}
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


@inline rand_vec_mf64(::Val{M}, ::Val{N}) where {M,N} =
    alternating_normalize(ntuple(_ -> rand_vec_f64(Val{M}()), Val{N}()))


################################################ MULTIFLOAT TEST CASE GENERATION


export prepare_mfadd_inputs, prepare_mfmul_inputs


function prepare_mfadd_inputs(x::NTuple{X,T}, y::NTuple{Y,T}) where {X,Y,T}
    return riffle(x, y)
end


@inline function two_prod(x::T, y::T) where {T}
    p = x * y
    e = fma(x, y, -p)
    return (p, e)
end


@generated function prepare_mfmul_inputs(
    x::NTuple{X,T},
    y::NTuple{Y,T},
) where {X,Y,T}
    tiers = Dict{Int,Vector{Tuple{Int,Int}}}()
    for i = 1:X
        for j = 1:Y
            tier = i + j
            if haskey(tiers, tier)
                push!(tiers[tier], (i, j))
            else
                tiers[tier] = [(i, j)]
            end
        end
    end
    xs = [Symbol('x', i) for i = 1:X]
    ys = [Symbol('y', j) for j = 1:Y]
    block = [
        Expr(:meta, :inline),
        Expr(:(=), Expr(:tuple, xs...), :x),
        Expr(:(=), Expr(:tuple, ys...), :y),
    ]
    ps = [Symbol('p', i, '_', j) for i = 1:X, j = 1:Y]
    es = [Symbol('e', i, '_', j) for i = 1:X, j = 1:Y]
    for i = 1:X
        for j = 1:Y
            push!(block, Expr(:(=),
                Expr(:tuple, ps[i, j], es[i, j]),
                Expr(:call, :two_prod, xs[i], ys[j])))
        end
    end
    result = Symbol[]
    for k = 2:X+Y
        for (i, j) in tiers[k]
            push!(result, ps[i, j])
        end
        for (i, j) in tiers[k]
            push!(result, es[i, j])
        end
    end
    push!(block, Expr(:return, Expr(:tuple, result...)))
    return Expr(:block, block...)
end


################################################################################

end # module TestCaseGenerators
