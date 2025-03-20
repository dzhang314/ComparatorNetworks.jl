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


function _inline_pass_expr(comparator, method::Symbol, N::Int)
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
    return _inline_pass_expr(C.instance, :forward, N)
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
    return _inline_pass_expr(C.instance, :backward, N)
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
    return _inline_pass_expr(C.instance, :riffle, N)
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


######################################################### CORRECTNESS PROPERTIES


export isbitsorted


@inline function isbitsorted(data::AbstractVector{T}) where {T}
    result = ~zero(T)
    @simd for i = firstindex(data):lastindex(data)-1
        @inbounds result &= data[i+1] | ~data[i]
    end
    return all(iszero(~result))
end


################################################################################

end # module Comparators
