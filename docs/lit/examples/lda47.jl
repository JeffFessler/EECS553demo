#=
# [LDA demo](@id lda47)

Illustrate
[Linear discriminant analysis (LDA)](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
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
using Plots: default, gui, savefig, twinx
using Plots: plot, plot!, scatter!, histogram!, RGB, cgrad
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
    digitn = [4,7] # which digits to use
    ## digitn = [1,7] # uncomment this for another case
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

digit_str = join(digitn); # string for file names
## savefig(pd, "lda$digit_str-digit.pdf")

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

# Sample mean training image:
dmean = sum(dtrain, dims = 3:4) / ntrain / ndigit
pm = jim(dmean; title="Mean image")

## savefig(pm, "lda$digit_str-mean.pdf")


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

## savefig(pu, "lda$digit_str-u.pdf")


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
function add_points!(p, X::AbstractArray{<:Number,3}; colors::Tuple = colors)
    for id in 1:ndigit
        scatter!(p, X[1,:,id], X[2,:,id];
            color = colors[id], label="$(digitn[id])")
    end
end
add_points!(petr, Xtrain)
add_points!(pete, Xtest1)
pp = plot(petr, pete; size = (950, 500))

## savefig(pp, "lda$digit_str-data.pdf")

#
prompt()

#=
## Illustrate maximum-likelihood estimates
=#

# Means
means = [vec(mean(X; dims=2)) for X in eachslice(Xtrain, dims=3)]
function plot_means!(p; dolabel::Bool = true)
    for id in 1:ndigit
        scatter!(p, [means[id][1]], [means[id][2]]; color = colors[id],
            markerstrokecolor = :black, marker = :square, markersize = 7,
            label = dolabel ? latexstring("\\mathbf{μ}_$(digitn[id])") : "",
        )
    end
    return p
end
plot_means!(petr)
plot!(petr, title = "Train data with sample means")
petr

#
prompt()

## savefig(petr, "lda$digit_str-means.pdf")


function add_ellipse!(p, mean, Σ; kwargs...)
    z = hcat([collect(sincos(t)) for t in range(0, 2π, 101)]...)
    xc = sqrt(2Σ) * z
    return plot!(p, mean[1] .+ xc[1,:], mean[2] .+ xc[2,:]; color=:black, kwargs...)
end;


# Plot de-meaned data:
Xdemean = cat([Xtrain[:,:,id] .- means[id] for id in 1:ndigit]..., dims=3)

pc = plot(; title = "Train data: de-meaned", args...)
add_points!(pc, Xdemean)

# Add covariance ellipse:
tmp = reshape(Xdemean, K, ntrain*ndigit) # pool all data
Σ = tmp * tmp' / size(tmp,2) # sample covariance

add_ellipse!(pc, zeros(2), Σ;
    label = "½ x'Σ⁻¹x = 1", legend = :topleft, legendfontsize = 12)

#
prompt()

## savefig(pc, "lda$digit_str-cov.pdf")


#=
## LDA classifier
brute force with argmax
=#

# LDA discriminant v1; brute-force way:
sqrtΣinv = inv(sqrt(Σ))
function lda_discriminant(
    x::AbstractVector;
    means::Vector{<:Vector} = means,
    sqrtΣinv::AbstractMatrix = sqrtΣinv,
    prob::AbstractVector = fill(1/ndigit, length(means)), # equal here
)
    score = copy(prob) # πₖ
    for id in 1:length(means)
        r = sqrtΣinv * (x - means[id]) # whitened residual
        score[id] *= exp(-(1/2) * sum(abs2, r))
    end
    return score ./ sum(score) # normalize by total probability
end


# LDA classifier v1; brute-force way:
function lda_classify1(x::AbstractVector; kwargs...)
    tmp = lda_discriminant(x; kwargs...)
    return digitn[argmax(tmp)]
end
#src lda_classify1([0,0]) # test


α = 0.2
color = cgrad([RGB(1-α, 1-α, 1), :black, RGB(1, 1-α, 1-α)])
x1_range = range(-6, 6, 221)
x2_range = range(-6, 6, 223)
function lda_plot(train_error::Real = NaN, test_error::Real = NaN;
    classifier::Function = lda_classify1,
    colorbar_ticks = digitn,
    clim = minmax(digitn),
    title::AbstractString =
        "LDA train error=$train_error %, test error = $test_error %",
)
    tmp = [classifier([x1; x2]) for x1 in x1_range, x2 in x2_range]
    p = jim(x1_range, x2_range, tmp;
        color, clim, title, prompt = false, colorbar_ticks, args...)
    add_points!(p, Xtrain)
    for id in 1:ndigit
        add_ellipse!(p, means[id], Σ)
    end
    plot_means!(p, dolabel = false)
    return p
end;


#=
## Classification errors
for train / validate / test
=#
function errors(data, label; classifier::Function = lda_classify1)
    data = reshape(data, K, :) # (d, n)
    err = count(classifier.(eachcol(data)) .!= label) / size(data, 2)
    return round(100 * err; sigdigits = 3)
end
train_error = errors(Xtrain, ytrain)
valid_error = errors(Xvalid, yvalid)
test1_error = errors(Xtest1, ytest1)
err1 = [ train_error valid_error test1_error ]


#=
## Plot data and decision regions
=#
p0 = lda_plot(train_error, test1_error;
    classifier = x -> lda_discriminant(x)[2], # show posterior probability
    colorbar_ticks = 0:0.5:1, clim = (0,1),
)

#
prompt()

## savefig(p0, "lda$digit_str-disc.pdf")
## savefig(p0, "lda$digit_str-v1.pdf")


#=
Histograms of discriminant values
=#

ph = plot(xlabel = L"⟨w,x⟩+b", ylabel = "count",
    title = "LDA Test data discriminant", legend = :topleft)
tmp = Vector{Any}(undef, ndigit)
ptmp = Vector{Any}(undef, ndigit)
for id in 1:ndigit
    function dfun(x) # inefficient way to get discriminant values!
       tmp = lda_discriminant(x)
       return log(tmp[2]/tmp[1])
    end
    tmp[id] = map(dfun, eachcol(Xtest1[:,:,id]))
    ptmp[id] = map(x -> lda_discriminant(x)[2], eachcol(Xtest1[:,:,id]))
    histogram!(ph, tmp[id], bins = 80,
        color = colors[id], linealpha = 0, linecolor = nothing, alpha = 0.5,
        label = "$(digitn[id])")
    scatter!(twinx(), tmp[id], ptmp[id]; color = :black, markersize = 2,
        yaxis = ("Posterior: P(Y=$(digitn[2]) ∣ x)", (0,1.02), 0:0.5:1))
end
ph


#=
Extension to the more efficient sign(⟨w,x⟩ + b) approach
is left as an exercise.
=#
