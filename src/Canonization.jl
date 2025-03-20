module Canonization

using ..ComparatorNetworks: ComparatorNetwork

################################################################################


export canonize


struct Instruction{T}
    opcode::Symbol
    outputs::Vector{T}
    inputs::Vector{T}
end


function _comparator_code(network::ComparatorNetwork{N}) where {N}
    generation = [1 for _ = 1:N]
    code = Instruction{Tuple{Int,Int}}[]
    for (x, y) in network.comparators
        x = Int(x)
        y = Int(y)
        @assert x < y
        gen_x = generation[x]
        gen_y = generation[y]
        outputs = [(x, gen_x + 1), (y, gen_y + 1)]
        inputs = [(x, gen_x), (y, gen_y)]
        push!(code, Instruction(:comparator, outputs, inputs))
        generation[x] = gen_x + 1
        generation[y] = gen_y + 1
    end
    outputs = [(i, generation[i]) for i = 1:N]
    inputs = [(i, 1) for i = 1:N]
    return (code, outputs, inputs)
end


function _is_valid_ssa(
    code::AbstractVector{Instruction{T}},
    outputs::AbstractVector{T},
    inputs::AbstractVector{T},
) where {T}
    computed = Set{T}(inputs)
    for instr in code
        for input in instr.inputs
            if !(input in computed)
                return false
            end
        end
        for output in instr.outputs
            if output in computed
                return false
            end
            push!(computed, output)
        end
    end
    return issubset(outputs, computed)
end


function _eliminate_dead_code!(
    code::AbstractVector{Instruction{T}},
    outputs::AbstractVector{T},
) where {T}
    needed = Set{T}(outputs)
    dead_indices = BitSet()
    for (index, instr) in Iterators.reverse(pairs(code))
        if any(output in needed for output in instr.outputs)
            for input in instr.inputs
                push!(needed, input)
            end
        else
            push!(dead_indices, index)
        end
    end
    deleteat!(code, dead_indices)
    return !isempty(dead_indices)
end


function _delete_duplicate_comparators!(
    code::AbstractVector{Instruction{T}},
    outputs::AbstractVector{T},
) where {T}
    compared = Dict{T,T}()
    identical = Dict{T,T}()
    dead_indices = BitSet()
    for (index, instr) in pairs(code)
        @assert instr.opcode == :comparator
        @assert length(instr.outputs) == 2
        @assert length(instr.inputs) == 2
        x_out, y_out = instr.outputs
        x_in, y_in = instr.inputs
        if ((haskey(compared, x_in) && (compared[x_in] == y_in)) ||
            (haskey(compared, y_in) && (compared[y_in] == x_in)))
            @assert (compared[x_in] == y_in) && (compared[y_in] == x_in)
            push!(dead_indices, index)
            identical[x_out] = x_in
            identical[y_out] = y_in
        else
            compared[x_out] = y_out
            compared[y_out] = x_out
        end
    end
    deleteat!(code, dead_indices)
    for instr in code
        for (index, input) in pairs(instr.inputs)
            if haskey(identical, input)
                instr.inputs[index] = identical[input]
            end
        end
        for (index, output) in pairs(instr.outputs)
            if haskey(identical, output)
                instr.outputs[index] = identical[output]
            end
        end
    end
    for (index, output) in pairs(outputs)
        if haskey(identical, output)
            outputs[index] = identical[output]
        end
    end
    return !isempty(dead_indices)
end


function _canonize_code!(
    code::AbstractVector{Instruction{T}},
    outputs::AbstractVector{T},
    inputs::AbstractVector{T},
) where {T}
    while true
        changed = false
        changed |= _eliminate_dead_code!(code, outputs)
        @assert _is_valid_ssa(code, outputs, inputs)
        changed |= _delete_duplicate_comparators!(code, outputs)
        @assert _is_valid_ssa(code, outputs, inputs)
        if !changed
            break
        end
    end
    if isempty(code)
        # Early return necessary to avoid calling `reduce(vcat, [])`.
        return code
    end
    generation = Dict{T,Int}()
    blocks = Vector{Vector{Instruction{T}}}()
    for input in inputs
        generation[input] = 0
    end
    for instr in code
        gen = 0
        for input in instr.inputs
            @assert haskey(generation, input)
            gen = max(gen, generation[input])
        end
        gen += 1
        if gen <= length(blocks)
            push!(blocks[gen], instr)
        else
            @assert gen == length(blocks) + 1
            push!(blocks, [instr])
        end
        for output in instr.outputs
            @assert !haskey(generation, output)
            generation[output] = gen
        end
    end
    for block in blocks
        sort!(block, by=instr -> instr.outputs)
    end
    return reduce(vcat, blocks)
end


function canonize(network::ComparatorNetwork{N}) where {N}
    code, outputs, inputs = _comparator_code(network)
    return ComparatorNetwork{N}([
        (UInt8(instr.outputs[1][1]), UInt8(instr.outputs[2][1]))
        for instr in _canonize_code!(code, outputs, inputs)])
end


################################################################################

end # module Canonization
