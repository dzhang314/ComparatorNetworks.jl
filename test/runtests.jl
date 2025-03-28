using ComparatorNetworks
using SIMD: Vec
using Test: @testset, @test

################################################################################


@testset "canonization" begin

    @test canonize(ComparatorNetwork{0}([])) == ComparatorNetwork{0}([])

    network_a = ComparatorNetwork{4}([
        (2, 3), (1, 4), (2, 3), (1, 4), (2, 4), (1, 3), (1, 3), (2, 4),
    ])
    network_b = ComparatorNetwork{4}([(1, 4), (2, 3), (1, 3), (2, 4)])
    @test canonize(network_a) == network_b
    @test canonize(network_b) == network_b

end


################################################################################


@testset "bitsort" begin

    SORTED_BIT_VECTORS_3 = (0x01, 0x17, 0x7F)

    SORTED_BIT_VECTORS_4 = (0x0001, 0x0117, 0x177F, 0x7FFF)

    SORTED_BIT_VECTORS_5 =
        (0x00000001, 0x00010117, 0x0117177F, 0x177F7FFF, 0x7FFFFFFF)

    SORTED_BIT_VECTORS_6 = (
        0x0000000000000001, 0x0000000100010117, 0x000101170117177F,
        0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF)

    SORTED_BIT_VECTORS_7 = (
        Vec{2,UInt64}((0x0000000000000000, 0x0000000000000001)),
        Vec{2,UInt64}((0x0000000000000001, 0x0000000100010117)),
        Vec{2,UInt64}((0x0000000100010117, 0x000101170117177F)),
        Vec{2,UInt64}((0x000101170117177F, 0x0117177F177F7FFF)),
        Vec{2,UInt64}((0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF)),
        Vec{2,UInt64}((0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF)),
        Vec{2,UInt64}((0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)))

    SORTED_BIT_VECTORS_8 = (
        Vec{4,UInt64}((
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000001)),
        Vec{4,UInt64}((
            0x0000000000000000, 0x0000000000000001,
            0x0000000000000001, 0x0000000100010117)),
        Vec{4,UInt64}((
            0x0000000000000001, 0x0000000100010117,
            0x0000000100010117, 0x000101170117177F)),
        Vec{4,UInt64}((
            0x0000000100010117, 0x000101170117177F,
            0x000101170117177F, 0x0117177F177F7FFF)),
        Vec{4,UInt64}((
            0x000101170117177F, 0x0117177F177F7FFF,
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF)),
        Vec{4,UInt64}((
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF,
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF)),
        Vec{4,UInt64}((
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)),
        Vec{4,UInt64}((
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)))

    SORTED_BIT_VECTORS_9 = (
        Vec{8,UInt64}((
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000001)),
        Vec{8,UInt64}((
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000001,
            0x0000000000000000, 0x0000000000000001,
            0x0000000000000001, 0x0000000100010117)),
        Vec{8,UInt64}((
            0x0000000000000000, 0x0000000000000001,
            0x0000000000000001, 0x0000000100010117,
            0x0000000000000001, 0x0000000100010117,
            0x0000000100010117, 0x000101170117177F)),
        Vec{8,UInt64}((
            0x0000000000000001, 0x0000000100010117,
            0x0000000100010117, 0x000101170117177F,
            0x0000000100010117, 0x000101170117177F,
            0x000101170117177F, 0x0117177F177F7FFF)),
        Vec{8,UInt64}((
            0x0000000100010117, 0x000101170117177F,
            0x000101170117177F, 0x0117177F177F7FFF,
            0x000101170117177F, 0x0117177F177F7FFF,
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF)),
        Vec{8,UInt64}((
            0x000101170117177F, 0x0117177F177F7FFF,
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF,
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF,
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF)),
        Vec{8,UInt64}((
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF,
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF,
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)),
        Vec{8,UInt64}((
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)),
        Vec{8,UInt64}((
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)))

    SORTED_BIT_VECTORS_10 = (
        Vec{16,UInt64}((
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000001)),
        Vec{16,UInt64}((
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000001,
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000001,
            0x0000000000000000, 0x0000000000000001,
            0x0000000000000001, 0x0000000100010117)),
        Vec{16,UInt64}((
            0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000001,
            0x0000000000000000, 0x0000000000000001,
            0x0000000000000001, 0x0000000100010117,
            0x0000000000000000, 0x0000000000000001,
            0x0000000000000001, 0x0000000100010117,
            0x0000000000000001, 0x0000000100010117,
            0x0000000100010117, 0x000101170117177F)),
        Vec{16,UInt64}((
            0x0000000000000000, 0x0000000000000001,
            0x0000000000000001, 0x0000000100010117,
            0x0000000000000001, 0x0000000100010117,
            0x0000000100010117, 0x000101170117177F,
            0x0000000000000001, 0x0000000100010117,
            0x0000000100010117, 0x000101170117177F,
            0x0000000100010117, 0x000101170117177F,
            0x000101170117177F, 0x0117177F177F7FFF)),
        Vec{16,UInt64}((
            0x0000000000000001, 0x0000000100010117,
            0x0000000100010117, 0x000101170117177F,
            0x0000000100010117, 0x000101170117177F,
            0x000101170117177F, 0x0117177F177F7FFF,
            0x0000000100010117, 0x000101170117177F,
            0x000101170117177F, 0x0117177F177F7FFF,
            0x000101170117177F, 0x0117177F177F7FFF,
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF)),
        Vec{16,UInt64}((
            0x0000000100010117, 0x000101170117177F,
            0x000101170117177F, 0x0117177F177F7FFF,
            0x000101170117177F, 0x0117177F177F7FFF,
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF,
            0x000101170117177F, 0x0117177F177F7FFF,
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF,
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF,
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF)),
        Vec{16,UInt64}((
            0x000101170117177F, 0x0117177F177F7FFF,
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF,
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF,
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF,
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF,
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF,
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)),
        Vec{16,UInt64}((
            0x0117177F177F7FFF, 0x177F7FFF7FFFFFFF,
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF,
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)),
        Vec{16,UInt64}((
            0x177F7FFF7FFFFFFF, 0x7FFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)),
        Vec{16,UInt64}((
            0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)))

    @test forward_fixed_point(bitminmax, all_bit_vectors(Val{3}())) === SORTED_BIT_VECTORS_3
    @test forward_fixed_point(bitminmax, all_bit_vectors(Val{4}())) === SORTED_BIT_VECTORS_4
    @test forward_fixed_point(bitminmax, all_bit_vectors(Val{5}())) === SORTED_BIT_VECTORS_5
    @test forward_fixed_point(bitminmax, all_bit_vectors(Val{6}())) === SORTED_BIT_VECTORS_6
    @test forward_fixed_point(bitminmax, all_bit_vectors(Val{7}())) === SORTED_BIT_VECTORS_7
    @test forward_fixed_point(bitminmax, all_bit_vectors(Val{8}())) === SORTED_BIT_VECTORS_8
    @test forward_fixed_point(bitminmax, all_bit_vectors(Val{9}())) === SORTED_BIT_VECTORS_9
    @test forward_fixed_point(bitminmax, all_bit_vectors(Val{10}())) === SORTED_BIT_VECTORS_10

    @test backward_fixed_point(bitminmax, all_bit_vectors(Val{3}())) === SORTED_BIT_VECTORS_3
    @test backward_fixed_point(bitminmax, all_bit_vectors(Val{4}())) === SORTED_BIT_VECTORS_4
    @test backward_fixed_point(bitminmax, all_bit_vectors(Val{5}())) === SORTED_BIT_VECTORS_5
    @test backward_fixed_point(bitminmax, all_bit_vectors(Val{6}())) === SORTED_BIT_VECTORS_6
    @test backward_fixed_point(bitminmax, all_bit_vectors(Val{7}())) === SORTED_BIT_VECTORS_7
    @test backward_fixed_point(bitminmax, all_bit_vectors(Val{8}())) === SORTED_BIT_VECTORS_8
    @test backward_fixed_point(bitminmax, all_bit_vectors(Val{9}())) === SORTED_BIT_VECTORS_9
    @test backward_fixed_point(bitminmax, all_bit_vectors(Val{10}())) === SORTED_BIT_VECTORS_10

    @test riffle_fixed_point(bitminmax, all_bit_vectors(Val{3}())) === SORTED_BIT_VECTORS_3
    @test riffle_fixed_point(bitminmax, all_bit_vectors(Val{4}())) === SORTED_BIT_VECTORS_4
    @test riffle_fixed_point(bitminmax, all_bit_vectors(Val{5}())) === SORTED_BIT_VECTORS_5
    @test riffle_fixed_point(bitminmax, all_bit_vectors(Val{6}())) === SORTED_BIT_VECTORS_6
    @test riffle_fixed_point(bitminmax, all_bit_vectors(Val{7}())) === SORTED_BIT_VECTORS_7
    @test riffle_fixed_point(bitminmax, all_bit_vectors(Val{8}())) === SORTED_BIT_VECTORS_8
    @test riffle_fixed_point(bitminmax, all_bit_vectors(Val{9}())) === SORTED_BIT_VECTORS_9
    @test riffle_fixed_point(bitminmax, all_bit_vectors(Val{10}())) === SORTED_BIT_VECTORS_10

end


################################################################################
