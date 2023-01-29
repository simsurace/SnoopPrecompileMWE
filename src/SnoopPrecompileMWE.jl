module SnoopPrecompileMWE

using KernelFunctions
using KernelFunctions: kappa, metric
using Zygote: gradient, @ignore

function KernelFunctions.kernelmatrix(
    κ::KernelFunctions.SimpleKernel, x::AbstractVector, y::AbstractVector
)
    KernelFunctions.validate_inputs(x, y)
    dist = metric(κ)
    return kappa.(Ref(κ), dist.(x, y'))
end

using SnoopPrecompile

struct Loss{Tx, Ty}
    x::Tx
    y::Ty
end

function (l::Loss)(θ)
    k = θ.variance * SEKernel() ∘ ScaleTransform(θ.lengthscale)
    return sum(kernelmatrix(k, l.x, l.y))
end

# This function compiles when the precompilation block below is commented out,
# but it generates an error 
# "ERROR: LoadError: Mutating arrays is not supported"
# when called within the precompilation block.
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
