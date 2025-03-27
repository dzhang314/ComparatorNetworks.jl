module MultiFloatError

using SIMD: Vec

using ..ComparatorNetworks: ComparatorNetwork, _unsafe_run_comparator_network!,
    two_sum, rand_vec_mf64, prepare_mfadd_inputs, prepare_mfmul_inputs

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


#=

export mfadd_relative_error, optimize_mfadd_relative_error,
    find_worst_case_mfadd_inputs

################################################################################


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

=#


################################################################################

end # module MultiFloatError
