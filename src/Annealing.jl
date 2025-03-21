module Annealing

using ..ComparatorNetworks: ComparatorNetwork, depth, AbstractCondition,
    _test_conditions, _random_comparator, generate_comparator_network

################################################################################


export mutate_comparator_network!, pareto_dominates, pareto_push!,
    optimize_comparator_network


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


@inline function pareto_dominates(s::T, t::T) where {T}
    all_le = true
    any_lt = false
    @inbounds for i in eachindex(s, t)
        all_le &= (s[i] <= t[i])
        any_lt |= (s[i] < t[i])
    end
    return all_le & any_lt
end


function pareto_push!(
    dict::AbstractDict{K,<:AbstractSet{V}},
    key::K,
    value::V,
) where {K,V}
    if haskey(dict, key)
        push!(dict[key], copy(value))
    else
        worse_keys = K[]
        for other_key in keys(dict)
            if pareto_dominates(other_key, key)
                return dict
            elseif pareto_dominates(key, other_key)
                push!(worse_keys, other_key)
            end
        end
        for worse_key in worse_keys
            delete!(dict, worse_key)
        end
        dict[key] = Set((copy(value),))
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
    temp_network = copy(network)
    initial_score = scoring_function(temp_network)
    result = Dict{typeof(initial_score),Set{ComparatorNetwork{N}}}()
    pareto_push!(result, initial_score, copy(temp_network))
    t = 1
    while true
        insert_weight = max(1, ceil(Int, insert_rate_function(t)))
        delete_weight = max(1, ceil(Int, delete_rate_function(t)))
        swap_weight = max(1, ceil(Int, swap_rate_function(t)))
        elapsed_ns = time_ns() - start_ns
        if elapsed_ns >= duration_ns
            return result
        end
        remaining_ns = duration_ns - elapsed_ns
        mutate_comparator_network!(temp_network, conditions...;
            insert_weight, delete_weight, swap_weight, duration_ns=remaining_ns)
        pareto_push!(result, scoring_function(temp_network), temp_network)
        t += 1
    end
end


function optimize_comparator_network(
    conditions::AbstractCondition{N}...;
    scoring_function=_default_scoring_function,
    insert_rate_function=_default_insert_rate_function,
    delete_rate_function=sqrt,
    swap_rate_function=log,
    duration_ns::Integer,
) where {N}
    start_ns = time_ns()
    temp_network = generate_comparator_network(conditions...)
    initial_score = scoring_function(temp_network)
    result = Dict{typeof(initial_score),Set{ComparatorNetwork{N}}}()
    pareto_push!(result, initial_score, copy(temp_network))
    elapsed_ns = time_ns() - start_ns
    if elapsed_ns >= duration_ns
        return result
    end
    return optimize_comparator_network(temp_network, conditions...;
        scoring_function,
        insert_rate_function, delete_rate_function, swap_rate_function,
        duration_ns=(duration_ns - elapsed_ns))
end


################################################################################

end # module Annealing
