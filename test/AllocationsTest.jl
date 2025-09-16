module AllocationsTest

using ForwardDiff

include(joinpath(dirname(@__FILE__), "utils.jl"))
DF1 = ForwardDiff.Dual{Nothing,Float64,8,Float64}
DF2 = ForwardDiff.Dual{Nothing,DF1,4,DF1}
convert_test_574() = convert(ForwardDiff.Dual{Nothing,DF2,2,DF2}, 1.3)

@testset "Test seed! allocations" begin
    x = rand(1000)
    cfg = ForwardDiff.GradientConfig(nothing, x)
    duals = cfg.duals
    seeds = cfg.seeds
    seed = cfg.seeds[1]

    alloc = @allocated ForwardDiff.seed!(duals, x, seeds)
    alloc = @allocated ForwardDiff.seed!(duals, x, seeds)
    @test alloc == 0

    alloc = @allocated ForwardDiff.seed!(duals, x, seed)
    alloc = @allocated ForwardDiff.seed!(duals, x, seed)
    @test alloc == 0

    index = 1
    alloc = @allocated ForwardDiff.seed!(duals, x, index, seeds)
    alloc = @allocated ForwardDiff.seed!(duals, x, index, seeds)
    @test alloc == 0

    index = 1
    alloc = @allocated ForwardDiff.seed!(duals, x, index, seed)
    alloc = @allocated ForwardDiff.seed!(duals, x, index, seed)
    @test alloc == 0

    alloc = @allocated convert_test_574()
    alloc = @allocated convert_test_574()
    @test_broken alloc == 0
    @test isbits(convert_test_574())

end

end
