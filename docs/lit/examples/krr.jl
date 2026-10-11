#=
# [Kernel ridge regression](@id krr)

This example illustrates
kernel ridge regression
using the Julia language.
=#

#srcURL

#=
## Setup
Add the Julia packages used in this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "InteractiveUtils"
        "LaTeXStrings"
        "LinearAlgebra"
        "MIRTjim"
        "Plots"
        "Random"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: norm, I
using MIRTjim: prompt
using Plots: default, gui, plot, plot!, scatter, scatter!, savefig
using Random: seed!
default(); default(label="", markerstrokecolor=:auto, widen=true, linewidth=2,
    markersize = 6, tickfontsize=12, labelfontsize = 16, legendfontsize=12)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


# ## Simulated data from latent nonlinear function
s = (t) -> atan(4*(t-0.5)) # nonlinear function

seed!(0) # seed rng
M = 15 # how many data points
xi = sort(rand(M)) # M random sample locations
y = s.(xi) + 0.1 * randn(M) # noisy samples

t0 = range(0, 1, 101) # fine sampling for showing curve
xaxis = (L"x", (0,1), 0:0.5:1)
yaxis = (L"y", (-1.3, 1.4), -1:1)
p0 = plot(t0, s.(t0); xaxis, yaxis,
 color=:blue, label="s(t) : latent signal", legend=:topleft)
scatter!(p0, xi, y, color=:black, label="y (noisy data)")

#
prompt()


#=
## Kernel ridge regression
=#

kernel(x,z,σ) = exp(-norm(x-z)^2/σ^2) # Gaussian kernel
σ = 5e-1 # chosen by trial and error
λ = 1e-2
Kmatrix = [kernel(a,b,σ) for a in xi, b in xi] # kernel matrix
w = (Kmatrix + M * λ * I) \ y # weights
regress(x) = w' * [kernel(xi,x,σ) for xi in xi]
yh = regress.(t0)
plot!(p0, t0, yh, label = "KRR prediction, λ=$λ, σ=$σ")

## savefig(p0, "krr-gauss.pdf")
