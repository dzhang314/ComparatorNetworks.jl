module Comparators

using SIMD: Vec, vifelse

########################################################### COMPARATOR FUNCTIONS


export bitminmax, two_sum, annihilating_maxmin


@inline bitminmax(x::T, y::T) where {T} = (x & y, x | y)


@inline function two_sum(x::T, y::T) where {T}
    s = x + y
    x_prime = s - y
    y_prime = s - x_prime
    delta_x = x - x_prime
    delta_y = y - y_prime
    e = delta_x + delta_y
    return (s, e)
end


@inline annihilating_maxmin(x::T, y::T) where {T} = (
    ifelse(x == y, zero(T), max(x, y)),
    ifelse(x == y, zero(T), min(x, y)))


@inline annihilating_maxmin(x::Vec{N,T}, y::Vec{N,T}) where {N,T} = (
    vifelse(x == y, zero(Vec{N,T}), max(x, y)),
    vifelse(x == y, zero(Vec{N,T}), min(x, y)))


############################################################## COMPARATOR PASSES


export forward_pass, forward_fixed_point,
    backward_pass, backward_fixed_point,
    riffle_pass, riffle_fixed_point


function _inline_pass_expr(comparator::Symbol, method::Symbol, N::Int)
    xs = [Symbol('x', i) for i in Base.OneTo(N)]
    body = Expr[]
    push!(body, Expr(:meta, :inline))
    push!(body, Expr(:(=), Expr(:tuple, xs...), :x))
    if method == :forward
        for i in 1:N-1
            push!(body, Expr(:(=), Expr(:tuple, xs[i], xs[i+1]),
                Expr(:call, comparator, xs[i], xs[i+1])))
        end
    elseif method == :backward
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


@generated function forward_pass(
    comparator::C,
    x::NTuple{N,T},
) where {N,T,C}
    return _inline_pass_expr(:comparator, :forward, N)
end


@inline function forward_fixed_point(
    comparator::C,
    x::NTuple{N,T},
) where {N,T,C}
    while true
        x_next = forward_pass(comparator, x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


@generated function backward_pass(
    comparator::C,
    x::NTuple{N,T},
) where {N,T,C}
    return _inline_pass_expr(:comparator, :backward, N)
end


@inline function backward_fixed_point(
    comparator::C,
    x::NTuple{N,T},
) where {N,T,C}
    while true
        x_next = backward_pass(comparator, x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


@generated function riffle_pass(
    comparator::C,
    x::NTuple{N,T},
) where {N,T,C}
    return _inline_pass_expr(:comparator, :riffle, N)
end


@inline function riffle_fixed_point(
    comparator::C,
    x::NTuple{N,T},
) where {N,T,C}
    while true
        x_next = riffle_pass(comparator, x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


################################################################# POSTCONDITIONS


export isbitsorted, IsCoarselySorted, IsNormalized,
    RelativeErrorBound, WeakRelativeErrorBound, IsCorrectlyRounded


@inline function isbitsorted(data::AbstractVector{T}) where {T}
    result = ~zero(T)
    @simd for i = firstindex(data):lastindex(data)-1
        @inbounds result &= data[i+1] | ~data[i]
    end
    return all(iszero(~result))
end


struct IsCoarselySorted{M,N} end


@inline function (::IsCoarselySorted{M,N})(data) where {M,N}
    @assert 1 <= M <= N
    Base.require_one_based_indexing(data)
    @assert length(data) == N
    first_item = @inbounds data[1]
    result = (first_item == first_item)
    prev_item = first_item
    for i = 2:M
        item = @inbounds data[i]
        result &= (iszero(prev_item) & iszero(item)) | (prev_item > item)
        prev_item = item
    end
    TM = eltype(data)(M)
    for i = M+1:N
        item = @inbounds data[i]
        result &= iszero(item) | (first_item >= item + TM)
    end
    return all(result)
end


struct IsNormalized{M,N} end


@inline function (cond::IsNormalized{M,N})(data) where {M,N}
    @assert 1 <= M <= N
    Base.require_one_based_indexing(data)
    @assert length(data) == N
    first_item = @inbounds data[1]
    result = (first_item == first_item)
    prev_item = first_item
    for i = 2:M
        item = @inbounds data[i]
        s, e = two_sum(prev_item, item)
        result &= ((s == prev_item) & (e == item)) |
                  (!isfinite(s)) | (!isfinite(e))
        prev_item = item
    end
    return all(result)
end


struct RelativeErrorBound{M,N,T}
    epsilon::T
end


@inline function (cond::RelativeErrorBound{M,N,T})(data) where {M,N,T}
    @assert 1 <= M <= N
    Base.require_one_based_indexing(data)
    @assert length(data) == N
    first_item = @inbounds data[1]
    result = (first_item == first_item)
    prev_item = first_item
    for i = 2:M
        item = @inbounds data[i]
        s, e = two_sum(prev_item, item)
        result &= ((s == prev_item) & (e == item)) |
                  (!isfinite(s)) | (!isfinite(e))
        prev_item = item
    end
    if !all(result)
        return false
    end
    tail = riffle_fixed_point(two_sum,
        ntuple(i -> (@inbounds data[M+i]), Val{N - M}()))
    if isempty(tail)
        return true
    end
    first_tail = @inbounds tail[1]
    return all((abs(first_tail) <= cond.epsilon * abs(first_item)) |
               (!isfinite(first_tail)) | (!isfinite(first_item)))
end


struct WeakRelativeErrorBound{M,N,T}
    epsilon::T
end


@inline function (cond::WeakRelativeErrorBound{M,N,T})(data) where {M,N,T}
    @assert 1 <= M <= N
    Base.require_one_based_indexing(data)
    @assert length(data) == N
    _EPS = eps(T)
    _WEAK_EPS = _EPS + _EPS
    _WEAKER_EPS = _WEAK_EPS + _WEAK_EPS
    first_item = @inbounds data[1]
    result = (first_item == first_item)
    prev_item = first_item
    for i = 2:M
        item = @inbounds data[i]
        result &= (abs(item) <= _WEAKER_EPS * abs(prev_item)) |
                  (!isfinite(item)) | (!isfinite(prev_item))
        prev_item = item
    end
    for i = M+1:N
        item = @inbounds data[i]
        result &= (abs(item) <= cond.epsilon * abs(first_item)) |
                  (!isfinite(item)) | (!isfinite(first_item))
    end
    return all(result)
end


struct IsCorrectlyRounded{M,N} end


@inline function (cond::IsCorrectlyRounded{M,N})(data) where {M,N}
    @assert 1 <= M <= N
    Base.require_one_based_indexing(data)
    @assert length(data) == N
    first_item = @inbounds data[1]
    result = (first_item == first_item)
    prev_item = first_item
    for i = 2:M
        item = @inbounds data[i]
        s, e = two_sum(prev_item, item)
        result &= ((s == prev_item) & (e == item)) |
                  (!isfinite(s)) | (!isfinite(e))
        prev_item = item
    end
    if !all(result)
        return false
    end
    if M == N
        return true
    end
    tail = riffle_fixed_point(two_sum,
        ntuple(i -> (@inbounds data[M+i]), Val{N - M}()))
    first_tail = @inbounds tail[1]
    s, e = two_sum(prev_item, first_tail)
    result &= ((s == prev_item) & (e == first_tail)) |
              (!isfinite(s)) | (!isfinite(e))
    return all(result)
end


################################################################################


#=


export WeaklyNormalizedCondition


struct WeaklyNormalizedCondition{N,M} <: AbstractTwoSumCondition{N}
    function WeaklyNormalizedCondition{N,M}() where {N,M}
        @assert N >= M >= 1
        return new{N,M}()
    end
end


@inline _unsafe_exponent(x::T) where {T<:Base.IEEEFloat} = Int(
    (reinterpret(Unsigned, x) & ~Base.sign_mask(T)) >> Base.significand_bits(T)
) - Base.exponent_bias(T)


@inline function _is_weakly_normalized(a::T, b::T) where {T}
    if iszero(b)
        return true
    elseif iszero(a)
        return false
    else
        return _unsafe_exponent(a) >= _unsafe_exponent(b) + precision(T)
    end
end


@inline function _is_weakly_normalized(a::T, b::T, n::Int) where {T}
    if iszero(b)
        return true
    elseif iszero(a)
        return false
    else
        return _unsafe_exponent(a) >= _unsafe_exponent(b) + n * precision(T)
    end
end


function (cond::WeaklyNormalizedCondition{N,M})(
    x::NTuple{N,T},
) where {N,M,T}
    @inbounds for i = 1:M-1
        if !_is_weakly_normalized(x[i], x[i+1])
            return false
        end
    end
    @inbounds first_limb = x[1]
    @inbounds for i = M+1:N
        if !_is_weakly_normalized(first_limb, x[i], M)
            return false
        end
    end
    return true
end


function (cond::WeaklyNormalizedCondition{N,M})(
    x::AbstractVector{T},
) where {N,M,T}
    Base.require_one_based_indexing(x)
    @assert length(x) == N
    @inbounds for i = 1:M-1
        if !_is_weakly_normalized(x[i], x[i+1])
            return false
        end
    end
    @inbounds first_limb = x[1]
    @inbounds for i = M+1:N
        if !_is_weakly_normalized(first_limb, x[i], M)
            return false
        end
    end
    return true
end


export IncompletelyNormalizedCondition


struct IncompletelyNormalizedCondition{N,M,K} <: AbstractTwoSumCondition{N}
    function IncompletelyNormalizedCondition{N,M,K}() where {N,M,K}
        @assert N >= M >= 1
        @assert K >= 0
        return new{N,M,K}()
    end
end


@inline function _is_incompletely_normalized(a::T, b::T, k::Int) where {T}
    if iszero(b)
        return true
    elseif iszero(a)
        return false
    else
        return _unsafe_exponent(a) >= _unsafe_exponent(b) + (precision(T) - k)
    end
end


@inline function _is_incompletely_normalized(
    a::T, b::T, n::Int, k::Int,
) where {T}
    if iszero(b)
        return true
    elseif iszero(a)
        return false
    else
        return (_unsafe_exponent(a) >=
                _unsafe_exponent(b) + n * (precision(T) - k))
    end
end


function (cond::IncompletelyNormalizedCondition{N,M,K})(
    x::NTuple{N,T},
) where {N,M,K,T}
    @inbounds for i = 1:M-1
        if !_is_incompletely_normalized(x[i], x[i+1], K)
            return false
        end
    end
    @inbounds first_limb = x[1]
    @inbounds for i = M+1:N
        if !_is_incompletely_normalized(first_limb, x[i], M, K)
            return false
        end
    end
    return true
end


function (cond::IncompletelyNormalizedCondition{N,M,K})(
    x::AbstractVector{T},
) where {N,M,K,T}
    Base.require_one_based_indexing(x)
    @assert length(x) == N
    @inbounds for i = 1:M-1
        if !_is_incompletely_normalized(x[i], x[i+1], K)
            return false
        end
    end
    @inbounds first_limb = x[1]
    @inbounds for i = M+1:N
        if !_is_incompletely_normalized(first_limb, x[i], M, K)
            return false
        end
    end
    return true
end


export StronglyNormalizedCondition


struct StronglyNormalizedCondition{N,M} <: AbstractTwoSumCondition{N}
    function StronglyNormalizedCondition{N,M}() where {N,M}
        @assert N >= M >= 1
        return new{N,M}()
    end
end


@inline _is_strongly_normalized(a::T, b::T) where {T} =
    (a, b) === two_sum(a, b)


function (cond::StronglyNormalizedCondition{N,M})(
    x::NTuple{N,T},
) where {N,M,T}
    @inbounds for i = 1:M-1
        if !_is_strongly_normalized(x[i], x[i+1])
            return false
        end
    end
    @inbounds final_limb = x[M]
    @inbounds for i = M+1:N
        if !_is_strongly_normalized(final_limb, x[i])
            return false
        end
    end
    return true
end


function (cond::StronglyNormalizedCondition{N,M})(
    x::AbstractVector{T},
) where {N,M,T}
    Base.require_one_based_indexing(x)
    @assert length(x) == N
    @inbounds for i = 1:M-1
        if !_is_strongly_normalized(x[i], x[i+1])
            return false
        end
    end
    @inbounds final_limb = x[M]
    @inbounds for i = M+1:N
        if !_is_strongly_normalized(final_limb, x[i])
            return false
        end
    end
    return true
end


=#


end # module Comparators
