module SnoopPrecompileMWE

using Distances
using Zygote: gradient, @ignore

using SnoopPrecompile

struct Loss{Tx, Ty}
    x::Tx
    y::Ty
end

function (l::Loss)(θ)
    dist = SqEuclidean()
    # The following line contains mutating code,
    # but does not need to be differentiated.
    # Without the @ignore, the precompilation directive will fail.
    D = @ignore pairwise(dist, l.x, l.y)
    return sum(θ.variance .* exp.(D ./ θ.lengthscale))
end

function compute_gradient(l, θ)
    return only(gradient(l, θ))
end

export Loss, compute_gradient

@precompile_setup begin
    l = Loss(randn(10), randn(10))
    θ = (variance = 1., lengthscale = 1.)

    @precompile_all_calls begin
        l(θ)
        compute_gradient(l, θ)
    end
end

end # module SnoopPrecompileMWE
