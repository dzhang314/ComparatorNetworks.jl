module ComparatorNetworks

using SIMD: Vec

################################################################################


export ComparatorNetwork


struct ComparatorNetwork{N}
    comparators::Vector{Tuple{UInt8,UInt8}}

    function ComparatorNetwork{N}(comparators::AbstractVector) where {N}
        @assert 0 <= N <= typemax(UInt8)
        my_comparators = Tuple{UInt8,UInt8}[]
        sizehint!(my_comparators, length(comparators))
        for (a, b) in comparators
            my_a = UInt8(a)
            @assert 0 < my_a <= N
            my_b = UInt8(b)
            @assert 0 < my_b <= N
            if my_a != my_b
                push!(my_comparators, minmax(my_a, my_b))
            end
        end
        return new{N}(my_comparators)
    end
end


@inline Base.length(network::ComparatorNetwork{N}) where {N} =
    length(network.comparators)
@inline Base.:(==)(a::ComparatorNetwork{N}, b::ComparatorNetwork{N}) where {N} =
    (a.comparators == b.comparators)
@inline Base.copy(network::ComparatorNetwork{N}) where {N} =
    ComparatorNetwork{N}(network.comparators)
@inline Base.hash(network::ComparatorNetwork{N}, h::UInt) where {N} =
    hash(network.comparators, h)


################################################################################


struct Instruction{T}
    opcode::Symbol
    outputs::Vector{T}
    inputs::Vector{T}
end


function sorting_code(network::ComparatorNetwork{N}) where {N}
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
        push!(code, Instruction(:minmax, outputs, inputs))
        generation[x] = gen_x + 1
        generation[y] = gen_y + 1
    end
    outputs = [(i, generation[i]) for i = 1:N]
    inputs = [(i, 1) for i = 1:N]
    return (code, outputs, inputs)
end


function accumulation_code(network::ComparatorNetwork{N}) where {N}
    generation = [1 for _ = 1:N]
    code = Instruction{Tuple{Int,Int}}[]
    k = N
    for (x, y) in network.comparators
        x = Int(x)
        y = Int(y)
        @assert x < y
        gen_x = generation[x]
        gen_y = generation[y]
        k += 1
        # TwoSum algorithm with temporary variables
        # k1 = x_prime, k2 = y_prime, k3 = delta_x, k4 = delta_y.
        push!(code, Instruction(:+, [(x, gen_x + 1)], [(x, gen_x), (y, gen_y)]))
        push!(code, Instruction(:-, [(k, 1)], [(x, gen_x + 1), (y, gen_y)]))
        push!(code, Instruction(:-, [(k, 2)], [(x, gen_x + 1), (k, 1)]))
        push!(code, Instruction(:-, [(k, 3)], [(x, gen_x), (k, 1)]))
        push!(code, Instruction(:-, [(k, 4)], [(y, gen_y), (k, 2)]))
        push!(code, Instruction(:+, [(y, gen_y + 1)], [(k, 3), (k, 4)]))
        generation[x] = gen_x + 1
        generation[y] = gen_y + 1
    end
    outputs = [(i, generation[i]) for i = 1:N]
    inputs = [(i, 1) for i = 1:N]
    return (code, outputs, inputs)
end


################################################################################


@inline _rand_vec_64() = Vec{8,UInt64}((
    rand(UInt64), rand(UInt64), rand(UInt64), rand(UInt64),
    rand(UInt64), rand(UInt64), rand(UInt64), rand(UInt64)))


@inline function _rand_vec_16()
    a = rand(UInt64)
    b = rand(UInt64)
    return Vec{8,UInt64}((
        a & 0xFFFF, (a >> 16) & 0xFFFF, (a >> 32) & 0xFFFF, a >> 48,
        b & 0xFFFF, (b >> 16) & 0xFFFF, (b >> 32) & 0xFFFF, b >> 48))
end


end # module ComparatorNetworks
