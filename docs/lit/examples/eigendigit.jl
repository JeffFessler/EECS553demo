#=
# [PCA for MNIST digits](@id eigendigit)

Illustrate
[Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
with MNIST hand-written digit images
in Julia.
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
        "MLDatasets"
        "Plots"
        "Statistics"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings: @L_str, latexstring
using LinearAlgebra: svd
using MIRTjim: jim, prompt
using MLDatasets: MNIST
using Plots: default, gui, savefig, RGB, cgrad, twinx
using Plots: plot, plot!, scatter, scatter!, histogram!, contour!
using Statistics: mean
default(); default(markersize=3, markerstrokecolor=:auto, label="",
 tickfontsize=14, labelfontsize=16, legendfontsize=16, titlefontsize=16)

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

#=
## Load data

Read the MNIST data for some handwritten digits.
This code will automatically download the data from web if needed
and put it in a folder like: `~/.julia/datadeps/MNIST/`.
=#
if !@isdefined(data)
    digitn = 0:9
    digit_str = join(digitn); # string for file names
    isinteractive() || (ENV["DATADEPS_ALWAYS_ACCEPT"] = true) # avoid prompt
    dataset = MNIST(Float32, :train)
    nrep = 1000 # how many of each digit
    ## function to extract the 1st `nrep` examples of digit n:
    data = n -> dataset.features[:,:,findall(==(n), dataset.targets)[1:nrep]]
    data = cat(dims=4, data.(digitn)...)
    labels = vcat([fill(d, nrep) for d in digitn]...) # to check later
    nx, ny, nrep, ndigit = size(data)
    data = data[:,2:ny,:,:] # make images non-square to force debug
    ny = size(data,2)
    size(data) # (nx, ny, nrep, ndigit)
end

# Look at some of the image data
pd = jim(data[:,:,1:10,:], "Data, d=$(nx*ny), N=$(nrep*ndigit)";
    colorbar = nothing, size = (600,500), tickfontsize = 6, ncol = 10)

## savefig(pd, "pca$digit_str-digit.pdf")

# Sample mean of all training images:
dmean = sum(data, dims = 3:4) / nrep / ndigit
pm = jim(dmean; title="Mean image")

## savefig(pm, "pca$digit_str-mean.pdf")

# Some pixels are always zero:
n0 = count(==(0), dmean)

#=
## PCA-based dimensionality reduction
=#
X = reshape(data .- dmean, nx*ny, :) # unfold
U, sigma, _ = svd(X);

# Show a few basis vectors
K = 20
tmp = reshape(U[:,1:K], nx, ny, K)
pu = jim(tmp; nrow=4, title="Basis functions, K=$K", color = :bwr,
    colorbar_ticks = [0])

## savefig(pu, "pca$digit_str-basis-$K.pdf")

#=
## Screen plot
=#

# non-log scale:
ps = scatter(sigma; widen = true,
   xaxis = ("k", (0, 756), [0:100:600; nx*ny-n0; nx*ny]),
   yaxis = (L"σ_k"),
   title = "Screen plot for digits $digit_str",
)


# log scale:
ps = scatter(log10.(sigma); widen = true,
   xaxis = ("k", (0, 756), [0:100:600; nx*ny-n0; nx*ny]),
   yaxis = (L"σ_k"),
   title = "Screen plot for digits $digit_str",
)


#=
## RMSE vs K
=#
total = sum(abs2, sigma) # total variance
rmse = sqrt.( (total .- cumsum(abs2.(sigma))) ./ total )
i01 = findfirst(≤(0.01), rmse)
scatter(rmse; widen = true,
   xaxis = (L"k", (0, 756), [0:100:400; i01; nx*ny-n0; nx*ny]),
   yaxis = ("RMSE", (0,1), ),
   title = "RMSE vs k for digits $digit_str",
)
