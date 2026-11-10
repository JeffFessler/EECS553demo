#=
# [Clustering demo](@id cluster1)

Illustrate
[k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)
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
        "Clustering"
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

using Clustering: kmeans
using InteractiveUtils: versioninfo
using LaTeXStrings: @L_str, latexstring
using LinearAlgebra: svd, det
using MIRTjim: jim, prompt
using MLDatasets: MNIST
using Plots: default, gui, savefig, RGB, cgrad, twinx
using Plots: plot, plot!, scatter!, histogram!, contour!
using Random: randperm, seed!
using Statistics: mean
default(); default(markersize=3, markerstrokecolor=:auto, label="",
 tickfontsize=14, labelfontsize=16, legendfontsize=16, titlefontsize=14)

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

#=
## Load data

Read the MNIST data for some handwritten digits.
This code will automatically download the data from web if needed
and put it in a folder like: `~/.julia/datadeps/MNIST/`.
=#
if !@isdefined(data) || true
    if !@isdefined(digitn)
        digitn = [0, 1, 7]
    end
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
    colorbar = nothing, size = (600,200), tickfontsize = 6, ncol = 10)

## savefig(pd, "km$digit_str-digit.pdf")


# Partition data into train / validate / test
ntrain = 200
nvalid = 100
ntest1 = 600
seed!(0)
tmp = randperm(nrep)
itrain = (1:ntrain)
ivalid = (1:nvalid) .+ ntrain
itest1 = (1:ntest1) .+ (ntrain + nvalid)
#src @assert sort([itrain; ivalid; itest1]) == 1:nrep
dtrain = data[:,:,tmp[itrain],:]
dvalid = data[:,:,tmp[ivalid],:]
dtest1 = data[:,:,tmp[itest1],:];

# Sample mean of all training images:
dmean = sum(dtrain, dims = 3:4) / ntrain / ndigit
pm = jim(dmean; title="Mean image")

## savefig(pm, "km$digit_str-mean.pdf")


#=
## PCA-based dimensionality reduction
Use two components for easy visualization
=#
Xdemean = reshape(dtrain .- dmean, nx*ny, :) # unfold
if !@isdefined(K)
    K = 2
end
U = svd(Xdemean).U[:,1:K];

# Show basis vectors
tmp = reshape(U, nx, ny, K)
pu = jim(tmp; nrow=1, title="Basis functions, K=$K", color = :bwr,
    size = (700, K==2 ? 400 : 150), colorbar_ticks = [0])

## savefig(pu, "km$digit_str-basis-$K.pdf")


#=
## Visualize embedded data
=#
embed(data, n) = reshape(U' * reshape(data .- dmean, nx*ny, :), K, n, :)
Xtrain = embed(dtrain, ntrain) # (K, ntrain, ndigit)
Xvalid = embed(dvalid, nvalid)
Xtest1 = embed(dtest1, ntest1)
labeler(n) = vcat([fill(digitn[id], n) for id in 1:ndigit]...)
ytrain = labeler(ntrain)
yvalid = labeler(nvalid)
ytest1 = labeler(ntest1)

xlim = (-9,6)
ylim = (-6,8)
args = (;
    xaxis = (L"x_1", xlim, -8:4:8),
    yaxis = (L"x_2", ylim, -8:4:8),
    aspect_ratio = 1,
    size = (550, 500),
)

petr = plot(; title = "PC data", args...)

colors = (:red, :green, :blue, :magenta, :orange)
for id in 1:ndigit
    scatter!(petr, Xtrain[1,:,id], Xtrain[2,:,id];
        label = "$(digitn[id])", color = colors[id])
end

X = reshape(Xtrain, K, ndigit * ntrain)
petb = plot(; title = "PC data", args...)
scatter!(petb, X[1,:], X[2,:]; color = :black)
pp = plot(petr, petb; size = (950, 500))

#
prompt()

## savefig(pp, "km$digit_str-embed.pdf")


function plot_cluster(ncluster::Int)
    out = kmeans(X, ncluster)

    p = plot(; title = "K-means K=$ncluster", args...)
    for k in 1:ncluster
        isk = out.assignments .== k
        scatter!(p, X[1,isk], X[2,isk];
            label = "$k", color = colors[k])
        ck = out.centers[:,k]
        scatter!(p, [ck[1]], [ck[2]]; markerstrokecolor = :black,
            marker = :square, color = colors[k], markersize = 6,)
    end
    return p
end

p2 = plot_cluster(2)
p3 = plot_cluster(3)
p4 = plot_cluster(4)
p5 = plot_cluster(5)

pk = plot(p2, p3, p4, p5;
    size = (950, 900),
)

## savefig(pk, "km$digit_str-kmeans.pdf")
