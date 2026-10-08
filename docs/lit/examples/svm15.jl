#=
# [SVM demo](@id svm15)

Illustrate
[SVM classification](https://en.wikipedia.org/wiki/Support_vector_machine)
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
        "ADTypes"
        "ForwardDiff"
        "InteractiveUtils"
        "LaTeXStrings"
        "LIBSVM"
        "LinearAlgebra"
        "MIRTjim"
        "MLDatasets"
        "Optim"
        "Plots"
        "Statistics"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

#src using ADTypes: AutoForwardDiff
import ForwardDiff
using InteractiveUtils: versioninfo
using LaTeXStrings: @L_str, latexstring
using LIBSVM: svmtrain, svmpredict, Kernel
using LinearAlgebra: svd, det
using MIRTjim: jim, prompt
using MLDatasets: MNIST
#src using Optim: optimize, LBFGS, minimizer
using Plots: default, gui, savefig, RGB, cgrad, twinx
using Plots: plot, plot!, scatter!, histogram!, contour!
using Random: randperm, seed!
using Statistics: mean
default(); default(markersize=3, markerstrokecolor=:auto, label="",
 tickfontsize=14, labelfontsize=16, legendfontsize=16, titlefontsize=16,
 colorbar_tickfontsize = 6, colorbar_titlefontsize = 14)

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

#=
## Load data

Read the MNIST data for some handwritten digits.
This code will automatically download the data from web if needed
and put it in a folder like: `~/.julia/datadeps/MNIST/`.
=#
if !@isdefined(data) # || true
    if !@isdefined(digitn)
        digitn = [1,5]
    end
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
    colorbar=nothing, size=(600,200), tickfontsize=6, ncol=10)

#
digit_str = join(digitn); # string for file names
## savefig(pd, "svm$digit_str-digit.pdf")

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

# Sample mean of training images:
dmean = sum(dtrain, dims = 3:4) / ntrain / ndigit
pm = jim(dmean; title="Mean image")

## savefig(pm, "svm$digit_str-mean.pdf")


#=
## PCA-based dimensionality reduction
Use two components for easy visualization
=#
X = reshape(dtrain .- dmean, nx*ny, :) # unfold
if !@isdefined(K)
    K = 2
end
U = svd(X).U[:,1:K];

# Show basis vectors
tmp = reshape(U, nx, ny, K)
pu = jim(tmp; nrow=1, title="Basis functions, K=$K", color = :bwr,
    size = (700, K==2 ? 400 : 150), colorbar_ticks = [0])

## savefig(pu, "svm$digit_str-basis-$K.pdf")


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

petr = plot(; title = "Train data", args...)
pete = plot(; title = "Test data", args...)

colors = (:blue, :red)
function add_points!(p, X::AbstractArray{<:Number,3}; colors::Tuple = colors)
    for id in 1:ndigit
        scatter!(p, X[1,:,id], X[2,:,id];
            color = colors[id], label="$(digitn[id])")
    end
end
add_points!(petr, Xtrain)
add_points!(pete, Xtest1)
pp = plot(petr, pete; size = (950, 500))

#
prompt()
## savefig(pp, "svm$digit_str-embed.pdf")


#=
## SVM classifier design
=#

kernel = Kernel.Linear; klabel = "Linear"; kernel_str = "linear"
kernel = Kernel.RadialBasis; klabel = "RadialBasis"; kernel_str = "rbf"
topm1(y) = y == digitn[1] ? 1 : -1
model = svmtrain(reshape(Xtrain, K, :), topm1.(ytrain); kernel);

# Support vectors
n_sv = length(model.SVs.indices)
sv = reshape(Xtrain, K, :)[:,model.SVs.indices]
#src w = only(eachcol(sv * model.coefs))
#src b = -only(model.rho) # ?

# Classifier helper
#src todo: [1] ?
#src svm_discriminant(x::AbstractVector) = kernel === Kernel.Linear ?
    #src w'x + b : svmpredict(model, reshape(x, :, 1))[2][1] + 0model.rho[1];
svm_discriminant(x::AbstractVector) =
    svmpredict(model, reshape(x, :, 1))[2][1] + 0model.rho[1];
#src svm_discriminant(zeros(K)) # test

todigit(y) = y == 1 ? digitn[1] : digitn[2]
svm_classify1(x::AbstractVector) =
    todigit(only(svmpredict(model, reshape(x, :, 1))[1]));
#src svm_classify1(zeros(K)) # test


#=
## Classification errors
for train / validate / test
=#
function errors(data, label; classifier::Function = svm_classify1)
    data = reshape(data, K, :) # (d, n)
    err = count(classifier.(eachcol(data)) .!= label) / size(data, 2)
    return round(100 * err; sigdigits = 3)
end
train_error = errors(Xtrain, ytrain)
valid_error = errors(Xvalid, yvalid)
test1_error = errors(Xtest1, ytest1)
err1 = [train_error valid_error test1_error]


#=
## Plot data and decision regions
(Only makes sense for K=2.)
=#
α = 0.6
color = reverse(cgrad([RGB(1-α, 1-α, 1), :black, RGB(1, 1-α, 1-α)])) # trick!
x1_range = range(-6, 6, 221)
x2_range = range(-6, 6, 223)
function svm_plot(train_error::Real = NaN, test_error::Real = NaN;
    classifier::Function = svm_classify1,
    title::AbstractString =
        "L.R. train error=$train_error %, test error = $test_error %",
)
    tmpc = [classifier([x1; x2; zeros(K-2)]) for x1 in x1_range, x2 in x2_range]
    tmpd = [svm_discriminant([x1; x2; zeros(K-2)]) for x1 in x1_range, x2 in x2_range]
@show size(tmpd) typeof(tmpd)

    p = jim(x1_range, x2_range, tmpd; color, title, prompt = false,
#src    clim = (0,1), colorbar_ticks = 0:0.5:1, # digitn,
#src    clim = tuple(digitn...), colorbar_ticks = digitn,
        colorbar_title = L"⟨w,Φ(x)⟩+b",
        annotate = (0, 5, "kernel = $kernel", :white),
        args...)
    plot!(p, colorbar = true)

    ## Add decision boundary
    contour!(p, x1_range, x2_range, tmpc'; color = :magenta)#, colorbar = :none)
    if kernel == Kernel.Linear
        contour!(p, x1_range, x2_range, abs.(tmpd)';
            color = :green, levels=[1e-3*maximum(tmpd)]) #, colorbar = :none
    end
    add_points!(p, Xtrain)
    return p
end;

p0 = svm_plot(train_error, test1_error);

# Mark support vectors
scatter!(p0, sv[1,:], sv[2,:], color = :white, marker = :x, alpha = 0.4,
    annotate = (-3, -5, "$n_sv / $(length(ytrain)) SVs", :yellow),
)

#
prompt()
## savefig(p0, "svm$digit_str-$kernel_str-v1.pdf")


#=
## Histograms of discriminant values
=#
ph = plot(xlabel = L"⟨w,Φ(x)⟩+b", ylabel = "count",
    title = "Test data discriminants, K=$K kernel=$kernel")
discs = Vector{Any}(undef, ndigit)
for id in 1:ndigit
    discs[id] = map(svm_discriminant, eachcol(Xtest1[:,:,id]))
    histogram!(ph, discs[id], bins = 80,
        color = colors[id], linealpha = 0, linecolor = nothing, alpha = 0.5,
        label = "$(digitn[id])")
end
ph

## savefig(ph, "svm$digit_str-$kernel_str-ph-$K.pdf")
