@time using SnoopPrecompileMWE 

## w/o precompile statements
# 2.568953 seconds (5.10 M allocations: 359.509 MiB, 5.17% gc time, 0.39% compilation time)

## w/ precompile statements, first time
# 17.620369 seconds (5.48 M allocations: 385.582 MiB, 0.77% gc time, 0.27% compilation time)

## w/ precompile statements, second time
# 2.794685 seconds (5.46 M allocations: 383.742 MiB, 5.05% gc time, 0.41% compilation time)

using KernelFunctions: RowVecs, ColVecs
losses = [
    Loss(randn(10), randn(10)),
    Loss(RowVecs(randn(10, 5)), RowVecs(randn(10, 5))),
    Loss(eachrow(randn(10, 5)), eachrow(randn(10, 5))),
    Loss(ColVecs(randn(10, 5)), ColVecs(randn(10, 5))),
    Loss(eachcol(randn(10, 5)), eachcol(randn(10, 5))),
    Loss([randn(5) for _ in  1:10], [randn(5) for _ in  1:10])
];
θ = (variance = 1., lengthscale = 1.);

for loss in losses
    @time loss(θ)
end

## w/o precompile statements
# 0.081053 seconds (289.35 k allocations: 19.437 MiB)
# 0.159219 seconds (570.63 k allocations: 38.681 MiB, 6.74% gc time)
# 0.130257 seconds (511.75 k allocations: 35.031 MiB, 8.63% gc time)
# 0.150489 seconds (525.82 k allocations: 35.878 MiB, 7.86% gc time)
# 0.115821 seconds (499.20 k allocations: 34.447 MiB)
# 0.066783 seconds (215.01 k allocations: 14.603 MiB, 11.55% gc time)

## w/ precompile statements
# 0.000013 seconds (4 allocations: 1008 bytes)
# 0.000039 seconds (2 allocations: 912 bytes)
# 0.000007 seconds (2 allocations: 912 bytes)
# 0.000005 seconds (2 allocations: 272 bytes)
# 0.000004 seconds (2 allocations: 272 bytes)
# 0.000005 seconds (4 allocations: 1008 bytes)

for loss in losses
    @time compute_gradient(loss, θ)
end

## w/o precompile statements
# 12.661393 seconds (24.57 M allocations: 1.553 GiB, 5.08% gc time)
# 1.160352 seconds (3.17 M allocations: 203.534 MiB, 9.06% gc time)
# 0.255233 seconds (920.16 k allocations: 58.944 MiB)
# 0.490022 seconds (1.40 M allocations: 90.359 MiB, 7.70% gc time)
# 0.253476 seconds (912.18 k allocations: 58.553 MiB)
# 0.401538 seconds (1.11 M allocations: 72.077 MiB, 11.36% gc time)

## w/ precompile statements
# 0.000291 seconds (198 allocations: 19.953 KiB)
# 0.009303 seconds (4.42 k allocations: 368.150 KiB)
# 0.000117 seconds (1.33 k allocations: 166.328 KiB)
# 0.000047 seconds (426 allocations: 59.594 KiB)
# 0.000044 seconds (426 allocations: 59.906 KiB)
# 0.000079 seconds (1.32 k allocations: 146.625 KiB)
