module MultiFloatError

using SIMD: Vec, vifelse

using ..ComparatorNetworks: ComparatorNetwork, _unsafe_run_comparator_network!,
    two_sum, riffle_fixed_point, rand_vec_mf64,
    prepare_mfadd_inputs, prepare_mfmul_inputs

############################################################ COMPARATOR PREFIXES


export commutative_mfadd_prefix, commutative_mfmul_prefix


function commutative_mfadd_prefix(X::Integer, Y::Integer)
    @assert X > 0
    @assert Y > 0
    return [(UInt8(2 * i - 1), UInt8(2 * i)) for i = 1:min(X, Y)]
end


commutative_mfadd_prefix(::Val{X}, ::Val{Y}) where {X,Y} =
    commutative_mfadd_prefix(X, Y)


function commutative_mfmul_prefix(X::Integer, Y::Integer)
    @assert X > 0
    @assert Y > 0
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
    terms = Tuple{Char,Int,Int}[]
    for k = 2:X+Y
        for (i, j) in tiers[k]
            push!(terms, ('p', i, j))
        end
        for (i, j) in tiers[k]
            push!(terms, ('e', i, j))
        end
    end
    mirror = Dict{Tuple{Char,Int,Int},Int}()
    for (k, (s, i, j)) in enumerate(terms)
        mirror[(s, j, i)] = k
    end
    result = Tuple{UInt8,UInt8}[]
    for (k, term) in enumerate(terms)
        if haskey(mirror, term)
            mk = mirror[term]
            if k < mk
                push!(result, (UInt8(k), UInt8(mk)))
            end
        end
    end
    return result
end


commutative_mfmul_prefix(::Val{X}, ::Val{Y}) where {X,Y} =
    commutative_mfmul_prefix(X, Y)


########################################################## COUNTEREXAMPLE SEARCH


export find_mfadd_counterexample, find_mfmul_counterexample


function find_mfadd_counterexample(
    ::Val{M},
    network::ComparatorNetwork{N},
    ::Val{X},
    ::Val{Y},
    postcondition::P,
    duration_ns::Integer,
) where {M,N,X,Y,P}
    # Assumes: network is well-formed (comparator indices lie in 1:N).
    start_ns = time_ns()
    @assert M >= 1
    @assert N == X + Y
    @assert X >= 1
    @assert Y >= 1
    @assert !signbit(duration_ns)
    temp_vector = Vector{Vec{M,Float64}}(undef, N)
    temp_scalar = Vector{Float64}(undef, N)
    count = 0
    while true
        x = rand_vec_mf64(Val{M}(), Val{X}())
        y = rand_vec_mf64(Val{M}(), Val{Y}())
        test_case_vector = prepare_mfadd_inputs(x, y)
        @simd ivdep for i = 1:N
            @inbounds temp_vector[i] = test_case_vector[i]
        end
        _unsafe_run_comparator_network!(temp_vector, network, two_sum)
        if !postcondition(temp_vector)
            for j = 1:M
                xj = ntuple(i -> (@inbounds x[i][j]), Val{X}())
                yj = ntuple(i -> (@inbounds y[i][j]), Val{Y}())
                test_case_scalar = prepare_mfadd_inputs(xj, yj)
                @simd ivdep for i = 1:N
                    @inbounds temp_scalar[i] = test_case_scalar[i][1]
                end
                _unsafe_run_comparator_network!(temp_scalar, network, two_sum)
                count += 1
                if !postcondition(temp_scalar)
                    return ((xj, yj), count)
                end
            end
            # This point should be unreachable. If the postcondition is
            # correctly implemented, then every vector counterexample should
            # contain a scalar counterexample.
            @assert false
        end
        count += M
        if time_ns() - start_ns >= duration_ns
            return (nothing, count)
        end
    end
end


function find_mfmul_counterexample(
    ::Val{M},
    network::ComparatorNetwork{N},
    ::Val{X},
    ::Val{Y},
    postcondition::P,
    duration_ns::Integer,
) where {M,N,X,Y,P}
    # Assumes: network is well-formed (comparator indices lie in 1:N).
    start_ns = time_ns()
    @assert M >= 1
    @assert N == 2 * X * Y
    @assert X >= 1
    @assert Y >= 1
    @assert !signbit(duration_ns)
    temp_vector = Vector{Vec{M,Float64}}(undef, N)
    temp_scalar = Vector{Float64}(undef, N)
    count = 0
    while true
        x = rand_vec_mf64(Val{M}(), Val{X}())
        y = rand_vec_mf64(Val{M}(), Val{Y}())
        test_case_vector = prepare_mfmul_inputs(x, y)
        @simd ivdep for i = 1:N
            @inbounds temp_vector[i] = test_case_vector[i]
        end
        _unsafe_run_comparator_network!(temp_vector, network, two_sum)
        if !postcondition(temp_vector)
            for j = 1:M
                xj = ntuple(i -> (@inbounds x[i][j]), Val{X}())
                yj = ntuple(i -> (@inbounds y[i][j]), Val{Y}())
                test_case_scalar = prepare_mfmul_inputs(xj, yj)
                @simd ivdep for i = 1:N
                    @inbounds temp_scalar[i] = test_case_scalar[i][1]
                end
                _unsafe_run_comparator_network!(temp_scalar, network, two_sum)
                count += 1
                if !postcondition(temp_scalar)
                    return ((xj, yj), count)
                end
            end
            # This point should be unreachable. If the postcondition is
            # correctly implemented, then every vector counterexample should
            # contain a scalar counterexample.
            @assert false
        end
        count += M
        if time_ns() - start_ns >= duration_ns
            return (nothing, count)
        end
    end
end


################################################## WORST CASE INPUT OPTIMIZATION


export find_worst_case_mfadd_inputs, find_worst_case_mfmul_inputs


@inline function _unsafe_multifloat_relative_error(
    data::AbstractVector{T},
    ::Val{M},
    ::Val{N},
) where {M,N,T}
    @assert 1 <= M <= N
    first_item = @inbounds data[1]
    prev_item = first_item
    for i = 2:M
        item = @inbounds data[i]
        s, e = two_sum(prev_item, item)
        if any(isfinite(s) & isfinite(e) & ((s != prev_item) | (e != item)))
            return one(T)
        end
        prev_item = item
    end
    if M == N
        return zero(T)
    end
    tail = riffle_fixed_point(two_sum,
        ntuple(i -> (@inbounds data[M+i]), Val{N - M}()))
    first_tail = @inbounds tail[1]
    return vifelse(iszero(first_tail), zero(T), abs(first_tail / first_item))
end


@inline function _unsafe_multifloat_relative_error!(
    temp::AbstractVector{T},
    network::ComparatorNetwork{N},
    input::NTuple{N,T},
    ::Val{Z},
) where {N,Z,T}
    @assert 1 <= Z <= N
    @simd ivdep for i = 1:N
        @inbounds temp[i] = input[i]
    end
    _unsafe_run_comparator_network!(temp, network, two_sum)
    return _unsafe_multifloat_relative_error(temp, Val{Z}(), Val{N}())
end


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


@inline function _unsafe_optimize_relative_error!(
    temp::AbstractVector{T},
    network::ComparatorNetwork{N},
    prepare_inputs,
    x::NTuple{X,T},
    y::NTuple{Y,T},
    ::Val{Z},
) where {N,X,Y,Z,T}
    x = riffle_fixed_point(two_sum, x)
    y = riffle_fixed_point(two_sum, y)
    max_error = _unsafe_multifloat_relative_error!(
        temp, network, prepare_inputs(x, y), Val{Z}())
    while true
        new_x = x
        new_y = y
        for i = 1:X*BITS_PER_BYTE*sizeof(T)
            flip_x = riffle_fixed_point(two_sum, _flip_bit(x, i - 1))
            if all(isfinite, flip_x)
                flip_error = _unsafe_multifloat_relative_error!(
                    temp, network, prepare_inputs(flip_x, new_y), Val{Z}())
                if flip_error > max_error
                    new_x = flip_x
                    max_error = flip_error
                end
            end
        end
        for i = 1:Y*BITS_PER_BYTE*sizeof(T)
            flip_y = riffle_fixed_point(two_sum, _flip_bit(y, i - 1))
            if all(isfinite, flip_y)
                flip_error = _unsafe_multifloat_relative_error!(
                    temp, network, prepare_inputs(new_x, flip_y), Val{Z}())
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


function find_worst_case_mfadd_inputs(
    ::Val{M},
    network::ComparatorNetwork{N},
    ::Val{X},
    ::Val{Y},
    ::Val{Z},
    duration_ns::Integer,
) where {M,N,X,Y,Z}
    # Assumes: network is well-formed (comparator indices lie in 1:N).
    start_ns = time_ns()
    @assert X + Y == N
    @assert 1 <= Z <= N
    max_error = zero(Float64)
    worst_case_x = ntuple(_ -> zero(Float64), Val{X}())
    worst_case_y = ntuple(_ -> zero(Float64), Val{Y}())
    count = 0
    temp_vector = Vector{Vec{M,Float64}}(undef, N)
    temp_scalar = Vector{Float64}(undef, N)
    while true
        x = rand_vec_mf64(Val{M}(), Val{X}())
        y = rand_vec_mf64(Val{M}(), Val{Y}())
        error_vector = _unsafe_multifloat_relative_error!(
            temp_vector, network, prepare_mfadd_inputs(x, y), Val{Z}())
        if any(isone(error_vector))
            for j = 1:M
                xj = ntuple(i -> (@inbounds x[i][j]), Val{X}())
                yj = ntuple(i -> (@inbounds y[i][j]), Val{Y}())
                error_scalar = _unsafe_multifloat_relative_error!(
                    temp_scalar, network, prepare_mfadd_inputs(xj, yj), Val{Z}())
                count += 1
                if isone(error_scalar)
                    return (1.0, (xj, yj), count)
                end
            end
            # This point should be unreachable. If the postcondition is
            # correctly implemented, then every vector counterexample should
            # contain a scalar counterexample.
            @assert false
        end
        if maximum(error_vector) > max_error
            for j = 1:M
                xj = ntuple(i -> (@inbounds x[i][j]), Val{X}())
                yj = ntuple(i -> (@inbounds y[i][j]), Val{Y}())
                error_scalar, opt_x, opt_y = _unsafe_optimize_relative_error!(
                    temp_scalar, network, prepare_mfadd_inputs,
                    xj, yj, Val{Z}())
                if error_scalar > max_error
                    max_error = error_scalar
                    worst_case_x = opt_x
                    worst_case_y = opt_y
                end
            end
        end
        count += M
        if time_ns() - start_ns >= duration_ns
            return (max_error, (worst_case_x, worst_case_y), count)
        end
    end
end


function find_worst_case_mfmul_inputs(
    ::Val{M},
    network::ComparatorNetwork{N},
    ::Val{X},
    ::Val{Y},
    ::Val{Z},
    duration_ns::Integer,
) where {M,N,X,Y,Z}
    # Assumes: network is well-formed (comparator indices lie in 1:N).
    start_ns = time_ns()
    @assert 2 * X * Y == N
    @assert 1 <= Z <= N
    max_error = zero(Float64)
    worst_case_x = ntuple(_ -> zero(Float64), Val{X}())
    worst_case_y = ntuple(_ -> zero(Float64), Val{Y}())
    count = 0
    temp_vector = Vector{Vec{M,Float64}}(undef, N)
    temp_scalar = Vector{Float64}(undef, N)
    while true
        x = rand_vec_mf64(Val{M}(), Val{X}())
        y = rand_vec_mf64(Val{M}(), Val{Y}())
        error_vector = _unsafe_multifloat_relative_error!(
            temp_vector, network, prepare_mfmul_inputs(x, y), Val{Z}())
        if any(isone(error_vector))
            for j = 1:M
                xj = ntuple(i -> (@inbounds x[i][j]), Val{X}())
                yj = ntuple(i -> (@inbounds y[i][j]), Val{Y}())
                error_scalar = _unsafe_multifloat_relative_error!(
                    temp_scalar, network, prepare_mfmul_inputs(xj, yj), Val{Z}())
                count += 1
                if isone(error_scalar)
                    return (1.0, (xj, yj), count)
                end
            end
            # This point should be unreachable. If the postcondition is
            # correctly implemented, then every vector counterexample should
            # contain a scalar counterexample.
            @assert false
        end
        if maximum(error_vector) > max_error
            for j = 1:M
                xj = ntuple(i -> (@inbounds x[i][j]), Val{X}())
                yj = ntuple(i -> (@inbounds y[i][j]), Val{Y}())
                error_scalar, opt_x, opt_y = _unsafe_optimize_relative_error!(
                    temp_scalar, network, prepare_mfmul_inputs,
                    xj, yj, Val{Z}())
                if error_scalar > max_error
                    max_error = error_scalar
                    worst_case_x = opt_x
                    worst_case_y = opt_y
                end
            end
        end
        count += M
        if time_ns() - start_ns >= duration_ns
            return (max_error, (worst_case_x, worst_case_y), count)
        end
    end
end


################################################################################

end # module MultiFloatError
