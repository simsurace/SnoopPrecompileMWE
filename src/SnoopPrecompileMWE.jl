module SnoopPrecompileMWE

using LinearAlgebra
using Zygote: gradient

struct Loss{Tx, Ty}
    x::Tx
    y::Ty
end

sqdist(x::Real, y::Real) = (x-y)^2
sqdist(x, y) = dot(x, x) + dot(y, y) - 2dot(x, y)
sekernel(v, l, x, y) = v * exp(-sqdist(x, y) / (2l^2))

function (l::Loss)(θ)
    # k = θ.variance * SEKernel() ∘ ScaleTransform(1/θ.lengthscale)
    # return sum(kernelmatrix(k, l.x, l.y))
    return sum(sekernel.(θ.variance, θ.lengthscale, l.x, permutedims(l.y)))
end

function compute_gradient(l, θ)
    return only(gradient(l, θ))
end

export Loss, compute_gradient

using SnoopPrecompile
using KernelFunctions: RowVecs, ColVecs

@precompile_setup begin
    losses = [
        Loss(randn(10), randn(10)),
        Loss(RowVecs(randn(10, 5)), RowVecs(randn(10, 5))),
        Loss(eachrow(randn(10, 5)), eachrow(randn(10, 5))),
        Loss(ColVecs(randn(10, 5)), ColVecs(randn(10, 5))),
        Loss(eachcol(randn(10, 5)), eachcol(randn(10, 5))),
        Loss([randn(5) for _ in  1:10], [randn(5) for _ in  1:10])
    ]
    θ = (variance = 1., lengthscale = 1.)

    @precompile_all_calls begin
        for loss in losses
            loss(θ)
            compute_gradient(loss, θ)
        end
    end
end

end # module SnoopPrecompileMWE
