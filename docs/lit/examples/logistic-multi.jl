#=
# [Multinomial logistic regression demo](@id logistic-multi)

Illustrate
[Multi-class logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)
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

using ADTypes: AutoForwardDiff
import ForwardDiff
using InteractiveUtils: versioninfo
using LaTeXStrings: @L_str, latexstring
using LinearAlgebra: svd, det
using MIRTjim: jim, prompt
using MLDatasets: MNIST
using Optim: optimize, LBFGS, minimizer
using Plots: default, gui, savefig, RGB, cgrad, twinx
using Plots: plot, plot!, scatter!, histogram!, contour!
using Random: randperm, seed!
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

## savefig(pd, "mlr$digit_str-digit.pdf")


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

## savefig(pm, "mlr$digit_str-mean.pdf")


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

## savefig(pu, "mlr$digit_str-basis-$K.pdf")


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

petr = plot(; title = "Train data", args...)
pete = plot(; title = "Test data", args...)

colors = (:red, :green, :blue)
for id in 1:ndigit
    scatter!(petr, Xtrain[1,:,id], Xtrain[2,:,id];
        label = "$(digitn[id])", color = colors[id])
    scatter!(pete, Xtest1[1,:,id], Xtest1[2,:,id];
        label = "$(digitn[id])", color = colors[id])
end
pp = plot(petr, pete; size = (950, 500))

#
prompt()

## savefig(pp, "mlr$digit_str-embed.pdf")


#=
## Helper functions
# softmax function and its log
=#
function log_softmax(x::AbstractVector{<:Number})
    shifted_x = x .- maximum(x)
    return shifted_x .- log.(sum(exp, shifted_x))
end
softmax(x::AbstractVector{<:Number}) = exp.(log_softmax(x));


#=
Cross-entropy between a 1-hot vector
(corresponding to an integer class label)
and the softmax of a discriminant vector
- `label` should be in `{1,…,ndigit}`
- `x` should be a vector of length `ndigit`
This corresponds to
(one term in)
the negative log-likelihood
used in multinomial logistic regression.
=#
function onehot_cross_entropy(label::Int, x::AbstractVector{<:Number})
   return -log_softmax(x)[label]
end;


#=
## Multinomial logistic classifier design
First set up the ERM cost function
(regularized negative log-likelihood).

- `vdata` is vector of data matrices that each should be d × nₖ
=#
function model_setup(vdata::AbstractVector{<:AbstractMatrix}, reg::Real)
    length(vdata) = ndigit || throw("dimension")
    ns = [size(X, 2) for X in vdata] # sample sizes for each class
    n = sum(ns) # total number of training samples
    Xbar = Vector{Any}(undef, ndigit)
    for id in 1:ndigit
        Xbar[id] = [ones(1, ns[id]); vdata[id]] # prepend 1
    end

    function cost(θ::AbstractVector)
        W = reshape(θ, ndigit, K+1) # θ = vec(W)
        total = 0
        for id in 1:ndigit
            tmp = W * Xbar[id] # discriminants
            fun = x -> onehot_cross_entropy(id, x)
            total += sum(fun, eachcol(tmp))
        end
        return (1/n) * total + reg/2 * sum(abs2, θ)
    end

    return cost
end;

# Regularization is essential because `W` is redundant
lambda = 1
erm_cost = model_setup(eachslice(Xtrain, dims=3), lambda);


#=
## Minimize cost
Use LBFGS quasi-Newton (QN) method.
The "L" is for limited memory
which is unimportant for this d=2 setting,
but is useful when applying QN
to the original data.
=#
W0 = zeros(ndigit, K+1) # todo: think about better init?
θ0 = vec(W0) # because `optimize` wants vectors
opt = optimize(erm_cost, θ0, LBFGS(); autodiff = AutoForwardDiff())
θhat = minimizer(opt)
What = reshape(θhat, size(W0))


# Multinomial logistic regression discriminant vector
function mlr_discriminant(x::AbstractVector; W::AbstractMatrix = What)
    return softmax(What * [1; x])
end;

# Logistic regression classifier that returns digit labels (not class index!)
function mlr_classify1(x::AbstractVector; kwargs...)
    return digitn[argmax(mlr_discriminant(x; kwargs...))]
end;

#src mlr_classify1(zeros(K)) # test


#=
## Classification errors
for train / validate / test
=#
function errors(data, label; classifier::Function = mlr_classify1)
    data = reshape(data, K, :) # (d, n)
    err = count(classifier.(eachcol(data)) .!= label) / size(data, 2)
    return round(100 * err; sigdigits = 3)
end
train_error = errors(Xtrain, ytrain)
valid_error = errors(Xvalid, yvalid)
test1_error = errors(Xtest1, ytest1)
err1 = [train_error valid_error test1_error]


#=
## Plot multinomial probabilities
(Only makes sense for K=2.)
=#
α = 0.7
x1_range = range(xlim..., 201)
x2_range = range(ylim..., 203)
tmp = [mlr_discriminant([x1; x2; zeros(K-2)]) for x1 in x1_range, x2 in x2_range]
tmp = [RGB(((1-α) .+ α*p)...) for p in tmp]

p0 = jim(x1_range, x2_range, tmp;
    title = "M.L.R. train error = $train_error %, test error = $test1_error %",
    prompt = false, xy_warn = false, args...,
)

for id in 1:ndigit
    scatter!(p0, Xtrain[1,:,id], Xtrain[2,:,id],
        color = colors[id], # markerstrokecolor = :black,
        label = "$(digitn[id])",
    )
end;


#=
## Plot decision boundaries
=#
tmp = [mlr_classify1([x1; x2; zeros(K-2)]) for x1 in x1_range, x2 in x2_range]
contour!(p0, x1_range, x2_range, tmp'; color = :black, colorbar = :none)

p0
## savefig(p0, "mlr$digit_str-softmax.pdf")

#
prompt()
