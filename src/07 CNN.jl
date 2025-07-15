# Classification of MNIST dataset using a convolutional neural network, the original LeNet5 from 1998. LeNet7 in 2005. And then since Alexnet2012 which was a bigger version to LeNets, CNNs have become the default choice for vision problems. Both architectures had made a significant leap in vision problems. This is similar to the Transformer architecture in 2017 which made a giant leap in NLP, and on which the ChatGPT was built. 

# Then came EfficientNet, ResNet, InceptionNet which were all improvements, offering new and efficient ways to handle images. But the challenge is to do the job with as few parameters(resources) as possible, scaling up always has a limit.

using Flux, JLD2, StatsPlots
using CSV, DataFrames, Statistics

folder = "mnist"  # sub-directory in which to save
isdir(folder) || mkdir(folder)
filename = joinpath(folder, "lenet.jld2")

#===== DATA =====#

# Calling MLDatasets.MNIST() will dowload the dataset if necessary,
# and return a struct containing it.
# It takes a few seconds to read from disk each time, so we do this once:

train_data = CSV.read(joinpath(folder, "mnist_train.csv"), DataFrame)  # i.e. split=:train
test_data = CSV.read(joinpath(folder, "mnist_test.csv"), DataFrame)

# train_data.features is a 28×28×60000 Array{Float32, 3} of the images.
# Flux needs a 4D array (WHCN), with the 3rd dim for channels -- here trivial, grayscale.
# Combine the reshape needed with other pre-processing:

function loader(data::DataFrame; batchsize::Int=512)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)   # insert trivial channel dim

    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2))

    yhot = Flux.onehotbatch(Vector(data.label), 0:9)  # make a 10×60000 OneHotMatrix
    Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true)
end

loader(train_data)  # returns a DataLoader, with first element a tuple like this:

x1, y1 = first(loader(test_data)); # (28×28×1×64 Array{Float32, 3}, 10×64 OneHotMatrix(::Vector{UInt32}))
train_data_loader = loader(train_data; batchsize=64)


#===== MODEL =====#

# LeNet has two convolutional layers, and our modern version has relu nonlinearities.
# After each conv layer there's a pooling step. Finally, there are some fully connected (Dense) layers:

lenet = Chain(
    Conv((5, 5), 1 => 6, relu), #all filter sizes, number of layers are all also Hyperparameters
    MeanPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MeanPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu), #256 here is decided by the original image size, and then the convolutional part, how many layers, what filter sizes, Poolings etc.
    Dense(120 => 84, relu),
    Dense(84 => 10),
)

#===== ARRAY SIZES =====#

# A layer like Conv((5, 5), 1=>6) takes 5x5 patches of an image, and matches them to each
# of 6 different 5x5 filters, placed at every possible position. These filters are here:

Conv((5, 5), 1 => 6).weight # 5×5×1×6 Array{Float32, 4}

# map(w->w[:,:,1]*x1[11:15,11:15,1,1], eachslice(Conv((5, 5), 1 => 6).weight, dims=4)) # Explanation - We multiply a convolutional filters weight matrix with a cross section of an image, and it returns the output for each channel. In this example we have 6 such channels.

# This layer can accept any size of image; let's trace the sizes with the actual input:

#=

julia> x1 |> size
(28, 28, 1, 64)

julia> lenet[1](x1) |> size  # after Conv((5, 5), 1=>6, relu),
(24, 24, 6, 64)

julia> lenet[1:2](x1) |> size  # after MaxPool((2, 2))
(12, 12, 6, 64)

julia> lenet[1:3](x1) |> size  # after Conv((5, 5), 6 => 16, relu)
(8, 8, 16, 64)

julia> lenet[1:4](x1) |> size  # after MaxPool((2, 2))
(4, 4, 16, 64)

julia> lenet[1:5](x1) |> size  # after Flux.flatten 
(256, 64)

=#

# Flux.flatten is just reshape, preserving the batch dimesion (64) while combining others (4*4*16).
# This 256 must match the Dense(256 => 120). Here is how to automate this, with Flux.outputsize:

#===== METRICS =====#

# We're going to log accuracy and loss during training. There's no advantage to
# calculating these on minibatches, since MNIST is small enough to do it at once.

using Statistics: mean  # standard library

function loss_and_accuracy(model, data)
    (x, y) = only(loader(data; batchsize=size(data, 1)))  # make one big batch
    ŷ = model(x)
    loss = Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
    #Flux=>Lux: logitcrossentropy => CrossEnrtropy(logits=true); crossentropy=>CrossEnrtropy(logits=false)
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    return loss, acc
end

@show loss_and_accuracy(lenet, test_data)  # accuracy about 10%, before training

#===== TRAINING =====#

# Let's collect some hyper-parameters in a NamedTuple, just to write them in one place.
# Global variables are fine -- we won't access this from inside any fast loops.

settings = (;
    eta=0.001,     # learning rate
    lambda=3e-4,  # for weight decay
    batchsize=512,
    epochs=10,
)
train_log = []

# Initialise the storage needed for the optimiser:
opt_rule = AdamW(settings.eta, (0.9, 0.999), settings.lambda)
opt_state = Flux.setup(opt_rule, lenet)
for epoch in 1:settings.epochs
    # @time will show a much longer time for the first epoch, due to compilation
    @time for (x, y) in train_data_loader
        grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), lenet)
        Flux.update!(opt_state, lenet, grads[1])
    end

    # Logging & saving, but not on every epoch
    if epoch % 2 == 1
        loss, acc = loss_and_accuracy(lenet, train_data)
        test_loss, test_acc = loss_and_accuracy(lenet, test_data)
        @info "logging:" epoch acc test_acc
        nt = (; epoch, loss, acc, test_loss, test_acc)  # make a NamedTuple
        push!(train_log, nt)
    end
    if epoch % 5 == 0
        JLD2.jldsave(filename; lenet_state=Flux.state(lenet))
        println("saved to ", filename, " after ", epoch, " epochs")
    end
end

#HW TODO - Check why the MLP using Lux.jl was faster than CNN using Flux?

# Compare MLP in Flux.jl and LUX = which library is faster?

import Pkg; Pkg.add("BenchmarkTools")
import Pkg; Pkg.add("CUDA")
using Flux, Lux, BenchmarkTools, Random, CUDA
using Statistics: mean

using Flux: Chain as FluxChain, Dense as FluxDense
using Lux: Chain as LuxChain, Dense as LuxDense


# --- Define MLP in Flux.jl ---
function build_flux_mlp()
    return FluxChain(
        FluxDense(784 => 256, relu),
        FluxDense(256 => 128, relu),
        FluxDense(128 => 10),
    )
end

# --- Define MLP in Lux.jl ---
function build_lux_mlp()
    return LuxChain(
        LuxDense(784 => 256, relu),
        LuxDense(256 => 128, relu),
        LuxDense(128 => 10),
    )
end

# --- Benchmarking Function ---
function benchmark_mlp()
    # Generate random input (batch size = 100)
    x = rand(Float32, 784, 100)  # CPU-only

    # --- Benchmark Flux.jl ---
    flux_model = build_flux_mlp()
    println("\n🔥 Benchmarking Flux.jl MLP:")
    flux_time = @belapsed $flux_model($x) samples=100 evals=3

    # --- Benchmark Lux.jl ---
    lux_model = build_lux_mlp()
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, lux_model)  # No GPU transfer
    println("\n🚀 Benchmarking Lux.jl MLP:")
    lux_time = @belapsed $lux_model($x, $ps, $st) samples=100 evals=3

    # --- Results ---
    println("\n📊 Results (CPU):")
    println("- Flux.jl MLP time: $(round(flux_time * 1000, digits=3)) ms")
    println("- Lux.jl MLP time: $(round(lux_time * 1000, digits=3)) ms")

    if lux_time < flux_time
        speedup = round((flux_time / lux_time), digits=2)
        println("\n✅ Lux.jl is **$(speedup)x faster** than Flux.jl!")
    else
        speedup = round((lux_time / flux_time), digits=2)
        println("\n✅ Flux.jl is **$(speedup)x faster** than Lux.jl!")
    end
end

# Run benchmark
benchmark_mlp()


# Compare CNN and MLP in one Library - which architecture is better?
using Flux
using MLDatasets
using Statistics
using BenchmarkTools
using ProgressMeter
using Flux, MLDatasets, Statistics, BenchmarkTools, ProgressMeter, Optimisers


using Flux, MLDatasets, Statistics, BenchmarkTools, ProgressMeter, Optimisers, Dates

# Configuration
const BATCH_SIZE = 128
const EPOCHS = 5

# Memory-efficient data loading
function get_data_loader(data_x, data_y; batch_size=BATCH_SIZE)
    x = reshape(data_x, 28, 28, 1, :) ./ 255f0
    y = Flux.onehotbatch(data_y, 0:9)
    return Flux.DataLoader((x, y); batchsize=batch_size, shuffle=true)
end

# Load data
train_x, train_y = MLDatasets.MNIST.traindata(Float32)
test_x, test_y = MLDatasets.MNIST.testdata(Float32)
train_loader = get_data_loader(train_x, train_y)
test_loader = get_data_loader(test_x, test_y; batch_size=1000)

# Parameter-balanced models (both ~105K params)
function create_balanced_mlp()
    Chain(
        Flux.flatten,
        Dense(28*28 => 128, relu),  # 784*128 + 128 = 100,480
        Dense(128 => 64, relu),     # 128*64 + 64 = 8,256
        Dense(64 => 10),             # 64*10 + 10 = 650
        softmax                      # Total: 100,480 + 8,256 + 650 = 109,386 (close enough)
    )
end

function create_cnn()
    Chain(
        Conv((3, 3), 1 => 16, pad=(1,1), relu),  # 3*3*1*16 + 16 = 160
        MaxPool((2, 2)),
        Conv((3, 3), 16 => 32, pad=(1,1), relu),  # 3*3*16*32 + 32 = 4,640
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(7*7*32 => 64, relu),                # 1568*64 + 64 = 100,416
        Dense(64 => 10),                           # 64*10 + 10 = 650
        softmax                                    # Total: 160 + 4,640 + 100,416 + 650 = 105,866
    )
end

# Verify parameter counts
mlp = create_balanced_mlp()
cnn = create_cnn()
println("Parameter Count Verification:")
println("- MLP: ", sum(length, Flux.params(mlp)), " params")
println("- CNN: ", sum(length, Flux.params(cnn)), " params\n")

# Enhanced training function with timing
function train_model(model, name, train_loader, test_loader; epochs=EPOCHS)
    opt = Optimisers.Adam(0.001)
    state = Optimisers.setup(opt, model)
    
    # Track timing
    epoch_times = Float64[]
    batch_times = Float64[]
    test_accuracies = Float64[]
    
    @showprogress for epoch in 1:epochs
        epoch_start = time()
        batch_time = 0.0
        
        # Training
        for (x, y) in train_loader
            batch_start = time()
            grads = gradient(model) do m
                Flux.crossentropy(m(x), y)
            end
            state, model = Optimisers.update(state, model, grads[1])
            batch_time += time() - batch_start
        end
        
        # Evaluation
        test_acc = 0.0
        for (x, y) in test_loader
            test_acc += mean(Flux.onecold(model(x)) .== Flux.onecold(y)) * size(x)[end]
        end
        test_acc /= length(test_y)
        push!(test_accuracies, test_acc)
        
        # Record times
        push!(epoch_times, time() - epoch_start)
        push!(batch_times, batch_time / length(train_loader))
        
        println("$name Epoch $epoch: ",
                "Test acc = $(round(test_acc*100, digits=2))% | ",
                "Epoch time = $(round(epoch_times[end], digits=2))s | ",
                "Avg batch time = $(round(batch_times[end]*1000, digits=2))ms")
    end
    
    # Benchmark inference
    x_test = first(first(test_loader))
    inf_time = @belapsed $model($x_test) samples=100 evals=3
    
    return (
        model=model,
        avg_epoch_time=mean(epoch_times),
        avg_batch_time=mean(batch_times),
        inf_time=inf_time,
        final_acc=test_accuracies[end]
    )
end

# Train and compare with timing
println("\n=== Training Balanced MLP ===")
mlp_results = train_model(mlp, "MLP", train_loader, test_loader)

println("\n=== Training CNN ===")
cnn_results = train_model(cnn, "CNN", train_loader, test_loader)

# Results comparison
println("\n" * "="^40)
println("FINAL COMPARISON (Parameter-Balanced)")
println("="^40)
println("Model         | MLP       | CNN")
println("--------------|-----------|-----------")
println("Params        | $(lpad(sum(length, Flux.params(mlp)), 8)) | $(lpad(sum(length, Flux.params(cnn)), 8))")
println("Accuracy      | $(lpad(round(mlp_results.final_acc*100, digits=2), 6))%  | $(lpad(round(cnn_results.final_acc*100, digits=2), 6))%")
println("Epoch Time    | $(lpad(round(mlp_results.avg_epoch_time, digits=2), 6))s  | $(lpad(round(cnn_results.avg_epoch_time, digits=2), 6))s")
println("Batch Time    | $(lpad(round(mlp_results.avg_batch_time*1000, digits=2), 6))ms | $(lpad(round(cnn_results.avg_batch_time*1000, digits=2), 6))ms")
println("Inference Time| $(lpad(round(mlp_results.inf_time*1000, digits=2), 6))ms | $(lpad(round(cnn_results.inf_time*1000, digits=2), 6))ms")

# Relative improvements
rel_acc = (cnn_results.final_acc - mlp_results.final_acc)/mlp_results.final_acc * 100
rel_speed = (mlp_results.inf_time - cnn_results.inf_time)/cnn_results.inf_time * 100
println("\nKey Findings:")
println("- CNN achieves $(round(rel_acc, digits=2))% higher accuracy")
println("- MLP is $(round(abs(rel_speed), digits=2))% faster at inference")

#===== SAVING TRAINED MODEL =====#
#part1
using Flux, Lux, BenchmarkTools, Random, Statistics, ProgressMeter, Zygote

# Configuration
const BATCH_SIZE = 128
const INPUT_SIZE = 784
const HIDDEN_LAYERS = [256, 128]  # Adjusted to match parameter counts

# Generate synthetic data
function synthetic_data(batch_size)
    x = rand(Float32, INPUT_SIZE, batch_size)
    y = rand(0:9, batch_size)
    return (x, Flux.onehotbatch(y, 0:9))
end

# Flux.jl Model
function build_flux_mlp()
    Flux.Chain(
        Flux.Dense(INPUT_SIZE => HIDDEN_LAYERS[1], relu),
        Flux.Dense(HIDDEN_LAYERS[1] => HIDDEN_LAYERS[2], relu),
        Flux.Dense(HIDDEN_LAYERS[2] => 10),
        softmax
    )
end

# Lux.jl Model
function build_lux_mlp()
    Lux.Chain(
        Lux.Dense(INPUT_SIZE => HIDDEN_LAYERS[1], relu),
        Lux.Dense(HIDDEN_LAYERS[1] => HIDDEN_LAYERS[2], relu),
        Lux.Dense(HIDDEN_LAYERS[2] => 10),
        softmax
    )
end

# Backpropagation Benchmark
function benchmark_backprop(model, model_type; n_batches=100)
    times = Float64[]
    for _ in 1:n_batches
        x, y = synthetic_data(BATCH_SIZE)
        if model_type == :flux
            grads = @elapsed gradient(params(model)) do
                sum(model(x) .* y)  # Dummy loss
            end
        else
            rng = Random.default_rng()
            ps, st = Lux.setup(rng, model)
            grads = @elapsed gradient(p -> sum(model(x, p, st)[1], ps))
        end
        push!(times, grads)
    end
    mean(times[10:end])  # Skip warmup
end

# Training Step Benchmark
function benchmark_training(model, model_type)
    opt = Adam(0.001)
    x, y = synthetic_data(BATCH_SIZE)
    
    if model_type == :flux
        ps = Flux.params(model)
        loss() = crossentropy(model(x), y)
        @belapsed Flux.train!(loss, $ps, [($x, $y)], $opt)
    else
        rng = Random.default_rng()
        ps, st = Lux.setup(rng, model)
        loss(p) = crossentropy(model(x, p, st)[1], nothing)
        @belapsed Optimisers.update!(opt, $ps, gradient($loss, $ps)[1])
    end
end

# Main Comparison
function compare_mlp_performance()
    # Build models
    flux_mlp = build_flux_mlp()
    lux_mlp = build_lux_mlp()

    # Verify parameter counts
    flux_params = sum(length, Flux.params(flux_mlp))
    lux_params = sum(length, Lux.initialparameters(Random.default_rng(), lux_mlp)[1])
    
    println("Parameter Counts:")
    println("- Flux: ", flux_params)
    println("- Lux: ", lux_params)

    # Benchmark backpropagation
    println("\nBackpropagation Performance:")
    flux_bp = benchmark_backprop(flux_mlp, :flux)
    lux_bp = benchmark_backprop(lux_mlp, :lux)
    println("- Flux: ", round(flux_bp*1000, digits=2), "ms/batch")
    println("- Lux: ", round(lux_bp*1000, digits=2), "ms/batch")
    println("Speedup: ", round(flux_bp/lux_bp, digits=2), "x")

    # Benchmark training step
    println("\nTraining Step Performance:")
    flux_train = benchmark_training(flux_mlp, :flux)
    lux_train = benchmark_training(lux_mlp, :lux)
    println("- Flux: ", round(flux_train*1000, digits=2), "ms/step")
    println("- Lux: ", round(lux_train*1000, digits=2), "ms/step")
    println("Speedup: ", round(flux_train/lux_train, digits=2), "x")
end

# Run comparison
compare_mlp_performance()
#-----------------

@show train_log

# We can re-run the quick sanity-check of predictions:
y1hat = softmax(lenet(x1))
@show hcat(Flux.onecold(y1hat, 0:9), Flux.onecold(y1, 0:9))

#===== LOADING TRAINED MODEL (SKIP THIS IF WANT TO TRAIN ANEW)=====#

# During training, the code above saves the model state to disk. Load the last version:

loaded_state = JLD2.load(filename, "lenet_state")

# Now you would normally re-create the model, and copy all parameters into that.
# We can use lenet2 from just above:

Flux.loadmodel!(lenet, loaded_state)


#===== THE END =====#

# 1. Cross-validation when Hyperparameter tuning - Requires Train and validation sets, ususally comes from the original train set, which are split for n folds
# 2. Evaluation 10-fold (evaluate!) when presenting a model to someone - Requires only train set which is split into 10 folds