#=
# [k-NN (k Nearest Neighbor) classifier demo](@id knn1)

Illustrate
[K-nearest neighbors classifier](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
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
        "LinearAlgebra"
        "MIRTjim"
        "MLDatasets"
        "NearestNeighbors"
        "NMF"
        "Plots"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: svd
using MIRTjim: jim, prompt
using MLDatasets: MNIST
using NearestNeighbors: KDTree, knn
using Plots: default, gui, savefig, plot, plot!, scatter!
using Random: randperm, seed!
default(); default(markersize=3, markerstrokecolor=:auto, label="",
 tickfontsize=14, labelfontsize=18, legendfontsize=18, titlefontsize=18)

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
    digitn = [4,7] # which digits to use
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
pd = jim(data[:,:,1:10,:], "Data, M=$(nx*ny), N=$(nrep*ndigit)";
    colorbar=nothing, size=(600,200), tickfontsize=6, ncol=10)

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

# mean training image
dmean = sum(dtrain, dims = 3:4) / ntrain / ndigit
pm = jim(dmean; title="Mean image")


#=
## PCA-based dimensionality reduction
Use two components for easy visualization
=#
X = reshape(dtrain .- dmean, nx*ny, :) # unfold
K = 2
U = svd(X).U[:,1:K];

# Show basis vectors
tmp = reshape(U, nx, ny, K)
pu = jim(tmp; title="Basis functions, K=$K", color=:cividis, size=(700,400))

#src savefig(pu, "knn-u.pdf")

#
prompt()


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

args = (;
 xaxis = (L"x_1", (-1,1) .* 6, -4:2:4),
 yaxis = (L"x_2", (-1,1) .* 6, -4:2:4),
 aspect_ratio = 1,
 size = (550, 500),
)

petr = plot(; title="Train data", args...)
pete = plot(; title="Test data", args...)

colors = (:blue, :red)
for id in 1:ndigit
    scatter!(petr, Xtrain[1,:,id], Xtrain[2,:,id], label="$(digitn[id])",
      color = colors[id])
    scatter!(pete, Xtest1[1,:,id], Xtest1[2,:,id], label="$(digitn[id])",
      color = colors[id])
end
pp = plot(petr, pete; size = (950, 500))

#
prompt()


#=
## Construct k-NN classifier
=#
data_tmp = reshape(Xtrain, K, ntrain*ndigit) # (K, ntrain)
tree = KDTree(data_tmp)

i2label = i -> getindex(ytrain, i)
function knn_class(point::AbstractVector, k::Int)
    idx, _ = knn(tree, point, k)
    labels = sort(i2label.(idx))
    return labels[k÷2+1] # majority vote
end

x1_range = range(-6, 6, 221)
x2_range = range(-6, 6, 223)

function knn_plot(k::Int)
    tmp = [knn_class([x1; x2], k) for x1 in x1_range, x2 in x2_range]
    p = jim(x1_range, x2_range, tmp;
        title = "k=$k", prompt = false, alpha = 0.4, args...)
    for id in 1:ndigit
        scatter!(p, Xtrain[1,:,id], Xtrain[2,:,id],
            color = colors[id],
            label = "$(digitn[id])",
        )
    end
    return p
end;

#=
## Decision regions for various k
=#

p1 = knn_plot(1)

#
p9 = knn_plot(9)

#
p25 = knn_plot(25)

#
p99 = knn_plot(99)


#=
## Train / validate / test accuracy
=#
klist = 1:30
data_tmp = reshape(Xtrain, K, ntrain*ndigit) # (d, n)
train_error = [count(knn_class.(eachcol(data_tmp), k) .!= ytrain) for k in klist] / (ntrain * ndigit) * 100
data_tmp = reshape(Xvalid, K, nvalid*ndigit) # (d, n)
valid_error = [count(knn_class.(eachcol(data_tmp), k) .!= yvalid) for k in klist] / (nvalid * ndigit) * 100
data_tmp = reshape(Xtest1, K, ntest1*ndigit) # (d, n)
test1_error = [count(knn_class.(eachcol(data_tmp), k) .!= ytest1) for k in klist] / (ntest1 * ndigit) * 100

pe = plot(
 xaxis = (L"k", (0,30), [1; argmin(valid_error); argmin(test1_error); 30]),
 yaxis = ("Error (%)", ),
)
plot!(klist, valid_error, marker=:x, label="Validation")
plot!(klist, test1_error, marker=:square, label="Test")
plot!(klist, train_error, marker=:o, label="Train")

## savefig(pe, "knn-error.pdf")


include("../../../inc/reproduce.jl")
