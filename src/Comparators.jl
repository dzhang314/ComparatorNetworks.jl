module Comparators

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
