#=
# [k-NN (k Nearest Neighbor) classifier demo](@id knn1)

[K-nearest neighbors classifier](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
of hand-written digit images
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
        "MLJModelInterface"
        "NearestNeighborModels"
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
using MLJModelInterface: fit, predict
using NearestNeighborModels: KNNClassifier
using Plots: default, gui, savefig, plot, scatter!
using Random: randperm, seed!
default(); default(markersize=5, markerstrokecolor=:auto, label="",
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

#
prompt()

# Partition data into train / validate / test
ntrain = 500
nvalid = 200
ntest1 = 300
seed!(0)
tmp = randperm(nrep)
itrain = (1:ntrain)
ivalid = (1:nvalid) .+ ntrain
itest1 = (1:ntest1) .+ (ntrain + nvalid)
#src @assert sort([itrain; ivalid; itest1]) == 1:nrep
dtrain = data[:,:,tmp[itrain],:]
dvalid = data[:,:,tmp[ivalid],:]
dtest1 = data[:,:,tmp[itest1],:]

# mean training image
dmean = sum(dtrain, dims = 3:4) / ntrain / ndigit
pm = jim(dmean; title="Mean image")


#=
## PCA-based dimensionality reduction
Use two components for easy visualization
=#
X = reshape(dtrain .- dmean, nx*ny, :) # unfold
K = 2
U = svd(X).U[:,1:K]

#=
## Basis vectors
=#
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

for id in 1:ndigit
    scatter!(petr, Xtrain[1,:,id], Xtrain[2,:,id], label="$(digitn[id])")
    scatter!(pete, Xtest1[1,:,id], Xtest1[2,:,id], label="$(digitn[id])")
end
plot(petr, pete; size = (950, 500))


#=
# everything below here is a WIP :(
    model_tmp = KNNClassifier(; K = 1)
    data_tmp = permutedims(reshape(Xtrain, K, :)) # (N, K)
    fit_tmp = fit(model_tmp, data_tmp, ytrain)

gui(); throw()

function knn_plot(k::Int)
# knn_plot(Xtrain_data::Matrix{Float64}, train_label::Vector{Int}, test_data::Matrix{Float64}, test_label::Vector{Int}; n_neighbors = 5)
    X_train_transposed = permutedims(train_data) # D x N
    X_test_transposed = permutedims(test_data)   # D x N

    model = KNNClassifier(; K = k)
    fitted_model = fit(model, X_train_transposed, train_label)

    x1_range = range(-3.0, stop=3.0, length=100)
    x2_range = range(-3.0, stop=3.0, length=100)

    xx = [x for x in x1_range, _ in x2_range]
    yy = [y for _ in x1_range, y in x2_range]

    grid_points = permutedims(hcat(vec(xx), vec(yy)))

    Z_pred_categorical = predict(fitted_model, grid_points)
    Z_pred_int = map(z -> parse(Int, string(z)), Z_pred_categorical)
    Z = reshape(Z_pred_int, size(xx))

    cmap_light = cgrad([colorant"#ADD8E6", colorant"#FFCCCB"])

    p1 = plot(
        framestyle=:box,
        background_color_subplot=:white,
        title="Decision Contours with Train Data and K value = $(n_neighbors)",
        xlabel="\$x_1\$", ylabel="\$x_2\$",
        xlims=(-3,3), ylims=(-3,3),
        legend=:topright,
        aspect_ratio=:equal
    )
    contourf!(p1, x1_range, x2_range, Z, levels=2, c=cmap_light, alpha=0.7, seriescolor=cmap_light, labels="")
    scatter!(p1, train_data[train_label.==1, 1], train_data[train_label.==1, 2],
             marker=:circle, color=:blue, label="+1", markersize=3)
    scatter!(p1, train_data[train_label.==2, 1], train_data[train_label.==2, 2],
             marker=:x, color=:red, label="-1", markersize=3)
    contour!(p1, x1_range, x2_range, Z, levels=[1.5], linecolor=:black, linewidth=1, labels="")


    p2 = plot(
        framestyle=:box,
        background_color_subplot=:white,
        title="Decision Contours with Test Data and K value = $(n_neighbors)",
        xlabel="\$x_1\$", ylabel="\$x_2\$",
        xlims=(-3,3), ylims=(-3,3),
        legend=:topright,
        aspect_ratio=:equal
    )
    contourf!(p2, x1_range, x2_range, Z, levels=2, c=cmap_light, alpha=0.7, seriescolor=cmap_light, labels="")
    scatter!(p2, test_data[test_label.==1, 1], test_data[test_label.==1, 2],
             marker=:circle, color=:blue, label="+1", markersize=3)
    scatter!(p2, test_data[test_label.==2, 1], test_data[test_label.==2, 2],
             marker=:x, color=:red, label="-1", markersize=3)
    contour!(p2, x1_range, x2_range, Z, levels=[1.5], linecolor=:black, linewidth=1, labels="")

    combined_plot = plot(p1, p2, layout=@layout([a b]), size=(1400, 700))

    ypred_test_categorical = predict(fitted_model, X_test_transposed)
    ypred_test_int = map(z -> parse(Int, string(z)), ypred_test_categorical)

    C_M_test = zeros(Int, 2, 2)
    for i in 1:length(test_label)
        true_idx = test_label[i]
        pred_idx = ypred_test_int[i]
        C_M_test[true_idx, pred_idx] += 1
    end

    false_positives = C_M_test[1, 2]
    false_negatives = C_M_test[2, 1]

    test_error_rate = ((false_positives + false_negatives) / sum(C_M_test)) * 100
    println("Test Error Rate for k=$(n_neighbors): $(round(test_error_rate, digits=2))%")

    return combined_plot
end
=#


include("../../../inc/reproduce.jl")
