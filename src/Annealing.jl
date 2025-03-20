module Annealing

using ..ComparatorNetworks: ComparatorNetwork, AbstractCondition,
    _random_comparator, generate_comparator_network

################################################################################


export pareto_dominates, pareto_push!, optimize_comparator_network


function _unsafe_mutate_comparator_network!(
    temp::AbstractVector{T},
    network::ComparatorNetwork{N},
    comparator::C,
    test_cases::AbstractVector{NTuple{N,T}},
    property::P,
    insert_weight::Int,
    delete_weight::Int,
    swap_weight::Int,
    duration_ns::UInt64,
) where {N,T,C,P}
    start_ns = time_ns()
    num_comparators = length(network.comparators)
    w1 = insert_weight
    w2 = w1 + delete_weight
    w3 = w2 + swap_weight
    while true
        w = rand(1:w3)
        if w <= w1 # insert
            i = rand(1:num_comparators+1)
            insert!(network.comparators, i, _random_comparator(Val{N}()))
            pass = _unsafe_test_comparator_network!(
                temp, network, comparator, test_cases, property)
            if pass
                return network
            end
            deleteat!(network.comparators, i)
        elseif (w <= w2) & (num_comparators >= 1) # delete
            i = rand(1:num_comparators)
            @inbounds original_comparator = network.comparators[i]
            deleteat!(network.comparators, i)
            pass = _unsafe_test_comparator_network!(
                temp, network, comparator, test_cases, property)
            if pass
                return network
            end
            insert!(network.comparators, i, original_comparator)
        elseif (w <= w3) & (num_comparators >= 2) # swap
            i = rand(1:num_comparators)
            j = rand(1:num_comparators-1)
            j += (j >= i)
            @inbounds network.comparators[i], network.comparators[j] =
                network.comparators[j], network.comparators[i]
            pass = _unsafe_test_comparator_network!(
                temp, network, comparator, test_cases, property)
            if pass
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


@inline function pareto_dominates(s, t)
    all_le = true
    any_lt = false
    @inbounds for i in eachindex(s, t)
        all_le &= (s[i] <= t[i])
        any_lt |= (s[i] < t[i])
    end
    return all_le & any_lt
end


function pareto_push!(dict::Dict{K,Set{V}}, key::K, value::V) where {K,V}
    if haskey(dict, key)
        push!(dict[key], value)
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
        dict[key] = Set((value,))
    end
    return dict
end


function _unsafe_anneal_comparator_network!(
    temp_data::AbstractVector{T},
    temp_network::ComparatorNetwork{N},
    comparator::C,
    test_cases::AbstractVector{NTuple{N,T}},
    property::P,
    scoring_function::S,
    insert_rate_function::F1,
    delete_rate_function::F2,
    swap_rate_function::F3,
    start_ns::UInt64,
    duration_ns::UInt64,
) where {N,T,C,P,S,F1,F2,F3}
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
        _unsafe_mutate_comparator_network!(
            temp_data, temp_network, comparator, test_cases, property,
            insert_weight, delete_weight, swap_weight, remaining_ns)
        pareto_push!(result, scoring_function(temp_network), copy(temp_network))
        t += 1
    end
end


function optimize_comparator_network(
    network::ComparatorNetwork{N},
    comparator::C,
    test_cases::AbstractVector{NTuple{N,T}},
    property::P,
    scoring_function::S,
    insert_rate_function::F1,
    delete_rate_function::F2,
    swap_rate_function::F3,
    duration_ns::UInt64,
) where {N,T,C,P,S,F1,F2,F3}
    start_ns = time_ns()
    temp_data = Vector{T}(undef, N)
    temp_network = copy(network)
    return _unsafe_anneal_comparator_network!(
        temp_data, temp_network,
        comparator, test_cases, property, scoring_function,
        insert_rate_function, delete_rate_function, swap_rate_function,
        start_ns, duration_ns)
end


function optimize_comparator_network(
    comparator::C,
    test_cases::AbstractVector{NTuple{N,T}},
    property::P,
    scoring_function::S,
    insert_rate_function::F1,
    delete_rate_function::F2,
    swap_rate_function::F3,
    duration_ns::UInt64,
) where {N,T,C,P,S,F1,F2,F3}
    start_ns = time_ns()
    temp_data = Vector{T}(undef, N)
    temp_network = _unsafe_generate_comparator_network!(
        temp_data, comparator, test_cases, property)
    return _unsafe_anneal_comparator_network!(
        temp_data, temp_network,
        comparator, test_cases, property, scoring_function,
        insert_rate_function, delete_rate_function, swap_rate_function,
        start_ns, duration_ns)
end


################################################################################

end # module Annealing
