module Annealing

using ..ComparatorNetworks: ComparatorNetwork, depth, canonize,
    AbstractCondition, _test_conditions, _random_comparator,
    generate_comparator_network

################################################################################


export mutate_comparator_network!, optimize_comparator_network


function mutate_comparator_network!(
    network::ComparatorNetwork{N},
    conditions::AbstractCondition{N}...;
    insert_weight::Integer,
    delete_weight::Integer,
    swap_weight::Integer,
    duration_ns::Integer,
) where {N}
    start_ns = time_ns()
    @assert !signbit(insert_weight)
    @assert !signbit(delete_weight)
    @assert !signbit(swap_weight)
    @assert !signbit(duration_ns)
    num_comparators = length(network.comparators)
    w1 = Int(insert_weight)
    w2 = Base.checked_add(w1, Int(delete_weight))
    w3 = Base.checked_add(w2, Int(swap_weight))
    while true
        w = rand(1:w3)
        if w <= w1 # insert
            i = rand(1:num_comparators+1)
            insert!(network.comparators, i, _random_comparator(Val{N}()))
            if _test_conditions(network, conditions...)
                return network
            end
            deleteat!(network.comparators, i)
        elseif (w <= w2) & (num_comparators >= 1) # delete
            i = rand(1:num_comparators)
            @inbounds original_comparator = network.comparators[i]
            deleteat!(network.comparators, i)
            if _test_conditions(network, conditions...)
                return network
            end
            insert!(network.comparators, i, original_comparator)
        elseif (w <= w3) & (num_comparators >= 2) # swap
            i = rand(1:num_comparators)
            j = rand(1:num_comparators-1)
            j += (j >= i)
            @inbounds network.comparators[i], network.comparators[j] =
                network.comparators[j], network.comparators[i]
            if _test_conditions(network, conditions...)
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


@inline function _pareto_le(s::T, t::T) where {T}
    all_le = true
    @inbounds for i in eachindex(s, t)
        all_le &= (s[i] <= t[i])
    end
    return all_le
end


@inline function _pareto_lt(s::T, t::T) where {T}
    all_le = true
    any_lt = false
    @inbounds for i in eachindex(s, t)
        all_le &= (s[i] <= t[i])
        any_lt |= (s[i] < t[i])
    end
    return all_le & any_lt
end


function _pareto_push!(
    dict::Dict{K,Dict{ComparatorNetwork{N},UInt64}},
    scoring_function,
    network::ComparatorNetwork{N},
    start_ns::UInt64,
) where {K,N}
    initial_key = scoring_function(network)
    if haskey(dict, initial_key) || any(
        _pareto_le(initial_key, key) for key in keys(dict))
        canonized = canonize(network)
        canonized_key = scoring_function(canonized)
        @assert _pareto_le(canonized_key, initial_key)
        if !haskey(dict, canonized_key)
            dict[canonized_key] = Dict{ComparatorNetwork{N},UInt64}()
            worse_keys = K[]
            for key in keys(dict)
                if _pareto_lt(canonized_key, key)
                    push!(worse_keys, key)
                end
            end
            for worse_key in worse_keys
                delete!(dict, worse_key)
            end
        end
        local_dict = dict[canonized_key]
        if !haskey(local_dict, canonized)
            local_dict[canonized] = time_ns() - start_ns
        end
    end
    return dict
end


@inline _default_scoring_function(network::ComparatorNetwork{N}) where {N} =
    (length(network), depth(network))


@inline _default_insert_rate_function(::Int) = 1


function optimize_comparator_network(
    network::ComparatorNetwork{N},
    conditions::AbstractCondition{N}...;
    scoring_function=_default_scoring_function,
    insert_rate_function=_default_insert_rate_function,
    delete_rate_function=sqrt,
    swap_rate_function=log,
    duration_ns::Integer,
) where {N}
    start_ns = time_ns()
    canonized = canonize(network)
    initial_score = scoring_function(canonized)
    result = Dict{typeof(initial_score),Dict{ComparatorNetwork{N},UInt64}}()
    result[initial_score] = Dict(canonized => zero(UInt64))
    temp_network = copy(network)
    for t = 1:typemax(Int)
        insert_weight = max(1, ceil(Int, insert_rate_function(t)))
        delete_weight = max(1, ceil(Int, delete_rate_function(t)))
        swap_weight = max(1, ceil(Int, swap_rate_function(t)))
        elapsed_ns = time_ns() - start_ns
        if elapsed_ns >= duration_ns
            return result
        end
        mutate_comparator_network!(temp_network, conditions...;
            insert_weight, delete_weight, swap_weight,
            duration_ns=(duration_ns - elapsed_ns))
        _pareto_push!(result, scoring_function, temp_network, start_ns)
        t += 1
    end
end


################################################################################

end # module Annealing
