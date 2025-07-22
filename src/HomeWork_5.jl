using Flux
using Flux: onehotbatch, onecold, DataLoader
using MLDatasets: CIFAR10
using Statistics: mean
using ProgressMeter
using Optimisers
import Pkg; Pkg.add("BSON")
using BSON
using Dates

# Create timestamped directory for results
results_dir = "results_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))"
mkpath(results_dir)

# 1. Load and prepare data
train_x, train_y = CIFAR10.traindata(Float32)
test_x, test_y = CIFAR10.testdata(Float32)

# Normalize and reshape data
normalize(x) = x ./ 255f0
train_x = normalize(reshape(train_x, 32, 32, 3, :))
test_x = normalize(reshape(test_x, 32, 32, 3, :))

# One-hot encode labels
train_y = onehotbatch(train_y, 0:9)
test_y = onehotbatch(test_y, 0:9)

# Create DataLoader
batch_size = 64
train_loader = DataLoader((train_x, train_y), batchsize=batch_size, shuffle=true)
test_loader = DataLoader((test_x, test_y), batchsize=batch_size)

# 2. Corrected LeNet-5 model for CIFAR-10 (32x32x3)
model = Chain(
    # Layer 1: Conv -> ReLU -> Pool
    Conv((5, 5), 3 => 6, relu, pad=(2, 2)),  # Output: 32x32x6
    MaxPool((2, 2)),                          # Output: 16x16x6
    
    # Layer 2: Conv -> ReLU -> Pool
    Conv((5, 5), 6 => 16, relu),             # Output: 12x12x16
    MaxPool((2, 2)),                          # Output: 6x6x16
    
    # Flatten and dense layers
    Flux.flatten,                             # Output: 6*6*16 = 576
    Dense(576 => 120, relu),                  # Corrected input size
    Dense(120 => 84, relu),
    Dense(84 => 10),
    softmax
)

# 3. Set up optimizer
optim = Optimisers.setup(Optimisers.Adam(0.001), model)

# 4. Define loss and accuracy
loss(x, y, model) = Flux.crossentropy(model(x), y)
accuracy(x, y, model) = mean(onecold(model(x)) .== onecold(y))

# Open a log file
log_file = open(joinpath(results_dir, "training_log.txt"), "w")
println(log_file, "Training Log - CIFAR10 with LeNet-5")
println(log_file, "=================================")
println(log_file, "Start Time: $(now())")
println(log_file, "Batch Size: $batch_size")
println(log_file, "Epochs: $epochs")
println(log_file, "Optimizer: Adam(0.001)")
println(log_file, "\nTraining Progress:")

# 5. Training loop
epochs = 10
test_accuracies = Float32[]
@showprogress for epoch in 1:epochs
    # Training phase
    for (x, y) in train_loader
        grads = Flux.gradient(m -> loss(x, y, m), model)
        optim, model = Optimisers.update(optim, model, grads[1])
    end
    
    # Evaluation phase
    test_acc = accuracy(test_x, test_y, model)
    push!(test_accuracies, test_acc)
    log_line = "Epoch $epoch: Test Accuracy = $(round(test_acc*100, digits=2))%"
    println(log_line)  # Print to console
    println(log_file, log_line)  # Write to log file
end

# Final evaluation
final_acc = accuracy(test_x, test_y, model)
final_log = "\nFinal Test Accuracy: $(round(final_acc*100, digits=2))%"
println(final_log)
println(log_file, final_log)
close(log_file)

# Save model weights
BSON.@save joinpath(results_dir, "model_weights.bson") model

# Save training curve data
using CSV
import DataFrames
using DataFrames
CSV.write(joinpath(results_dir, "training_curve.csv"), DataFrame(
    epoch = 1:epochs,
    accuracy = test_accuracies
))

# Save final metrics

using Pkg; Pkg.add("JSON")
using JSON
open(joinpath(results_dir, "training_metrics.json"), "w") do f
    JSON.print(f, Dict(
        "final_accuracy" => final_acc,
        "training_time" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
        "parameters" => Dict(
            "batch_size" => batch_size,
            "epochs" => epochs,
            "learning_rate" => 0.001
        )
    ), 4)
end

println("\nAll results saved to directory: $results_dir")


#######################################

""" Question2. Try to estimate the effect that new examples have on the performance of a neural network
compared to showing the same example several times. Train the network on a subset of 10000
examples for 6 epochs and evaluate the test set afterward. Now train on 20000 and 30000
examples while keeping the number of training steps identical i.e. 3 epoch for 20000 and 2
epochs for 30000 examples. Plot the final test accuracies and describe the observed behavior"""

    
using Flux, MLDatasets, Statistics, ProgressMeter, Plots
using Flux: onehotbatch, onecold, DataLoader
using MLDatasets: CIFAR10
using Random: shuffle

# 1. Correct Model Architecture
function build_model()
    Chain(
        # Block 1: 32x32x3 → 32x32x6 → 16x16x6
        Conv((5,5), 3 => 6, relu, pad=(2,2)),
        MaxPool((2,2)),
        
        # Block 2: 16x16x6 → 12x12x16 → 6x6x16
        Conv((5,5), 6 => 16, relu),
        MaxPool((2,2)),
        
        # Flatten: 6×6×16 = 576
        x -> reshape(x, :, size(x,4)),
        
        # Classifier
        Dense(576 => 120, relu),
        Dense(120 => 84, relu),
        Dense(84 => 10),
        softmax
    )
end

# 2. Data Loading
function load_data()
    train_x, train_y = CIFAR10.traindata(Float32)
    test_x, test_y = CIFAR10.testdata(Float32)
    
    μ = mean(train_x, dims=(1,2,4))
    σ = std(train_x, dims=(1,2,4))
    normalize(x) = (x .- μ) ./ max.(σ, 1f-6)
    
    train_x = normalize(reshape(train_x, 32, 32, 3, :))
    test_x = normalize(reshape(test_x, 32, 32, 3, :))
    
    return train_x, onehotbatch(train_y, 0:9),
           test_x, onehotbatch(test_y, 0:9)
end

# 3. Training Function
function train_model(loader, steps; lr=0.001)
    model = build_model()
    opt = Flux.setup(Adam(lr), model)
    
    for (i, (x,y)) in enumerate(loader)
        i > steps && break
        grads = gradient(model) do m
            Flux.crossentropy(m(x), y)
        end
        Flux.update!(opt, model, grads[1])
    end
    return model
end

# 4. Experiment Parameters
sizes = [10000, 20000, 30000]
epochs = [6, 3, 2]
base_batchsize = 64
base_steps = 937  # 10000/64*6 ≈ 937

# 5. Run Experiment
results = Float64[]
for (size, epoch) in zip(sizes, epochs)
    train_x, train_y, test_x, test_y = load_data()
    idxs = shuffle(1:50000)[1:size]
    
    batchsize = Int(ceil(size/(base_steps/epoch)))
    loader = DataLoader((train_x[:,:,:,idxs], train_y[:,idxs]),
                      batchsize=batchsize, shuffle=true)
    
    model = train_model(loader, base_steps)
    acc = mean(onecold(model(test_x)) .== onecold(test_y))
    push!(results, acc)
    
    # Correct rounding syntax:
    println("Size: $size | Epochs: $epoch | Batch: $batchsize → Acc: ", round(acc*100, digits=2), "%")
end

# 6. Plot Results
plot(sizes, results.*100,
    xlabel="Training Set Size",
    ylabel="Test Accuracy (%)",
    title="Dataset Size vs Performance (Constant Steps)",
    label="Accuracy",
    marker=:circle, linewidth=2)
savefig("dataset_size_effect.png")

""" Question 3. Measure the effect of filter size: Instead of using (5,5) filters in the LeNet5, use (3,3) filters
and (7,7) filters respectively to make a ”LeNet3” and ”LeNet7”. Plot the final test accuracies
and describe the observed behavior."""

using Flux, MLDatasets, Statistics, ProgressMeter, Plots
using Flux: onehotbatch, onecold, DataLoader
using MLDatasets: CIFAR10

# 1. Model Builders with verified dimensions
function build_lenet3()
    Chain(
        # Block 1: 32x32x3 → 32x32x6 → 16x16x6
        Conv((3,3), 3 => 6, relu, pad=1),  # pad=1 maintains 32x32
        MaxPool((2,2)),
        
        # Block 2: 16x16x6 → 16x16x16 → 8x8x16
        Conv((3,3), 6 => 16, relu, pad=1),  # pad=1 maintains 16x16
        MaxPool((2,2)),
        
        # Flatten: 8×8×16 = 1024
        x -> reshape(x, :, size(x,4)),
        Dense(1024 => 120, relu),  # Correct dimension
        Dense(120 => 84, relu),
        Dense(84 => 10),
        softmax
    )
end

function build_lenet5() 
    Chain(
        # Block 1: 32x32x3 → 32x32x6 → 16x16x6
        Conv((5,5), 3 => 6, relu, pad=2),  # pad=2 maintains 32x32
        MaxPool((2,2)),
        
        # Block 2: 16x16x6 → 12x12x16 → 6x6x16
        Conv((5,5), 6 => 16, relu),  # No padding reduces size
        MaxPool((2,2)),
        
        # Flatten: 6×6×16 = 576
        x -> reshape(x, :, size(x,4)),
        Dense(576 => 120, relu),  # Correct dimension
        Dense(120 => 84, relu),
        Dense(84 => 10),
        softmax
    )
end

function build_lenet7()
    Chain(
        # Block 1: 32x32x3 → 32x32x6 → 16x16x6
        Conv((7,7), 3 => 6, relu, pad=3),  # pad=3 maintains 32x32
        MaxPool((2,2)),
        
        # Block 2: 16x16x6 → 10x10x16 → 5x5x16
        Conv((7,7), 6 => 16, relu),  # No padding reduces size
        MaxPool((2,2)),
        
        # Flatten: 5×5×16 = 400
        x -> reshape(x, :, size(x,4)),
        Dense(400 => 120, relu),  # Correct dimension
        Dense(120 => 84, relu),
        Dense(84 => 10),
        softmax
    )
end

# 2. Training and Evaluation
function train_and_evaluate(model, name)
    # Load data
    train_x, train_y = CIFAR10.traindata(Float32)
    test_x, test_y = CIFAR10.testdata(Float32)
    
    # Normalize
    μ = mean(train_x, dims=(1,2,4))
    σ = std(train_x, dims=(1,2,4))
    normalize(x) = (x .- μ) ./ max.(σ, 1f-6)
    
    train_x = normalize(reshape(train_x, 32, 32, 3, :))
    test_x = normalize(reshape(test_x, 32, 32, 3, :))
    train_y = onehotbatch(train_y, 0:9)
    test_y = onehotbatch(test_y, 0:9)
    
    # Train
    loader = DataLoader((train_x, train_y), batchsize=64, shuffle=true)
    opt = Flux.setup(Adam(0.001), model)
    
    for epoch in 1:10
        for (x,y) in loader
            grads = gradient(model) do m
                Flux.crossentropy(m(x), y)
            end
            Flux.update!(opt, model, grads[1])
        end
    end
    
    # Evaluate
    acc = mean(onecold(model(test_x)) .== onecold(test_y))
    println("$name Accuracy: ", round(acc*100, digits=2), "%")
    return acc
end

# 3. Run comparison
results = Dict()
for (name, builder) in [("LeNet3", build_lenet3), 
                        ("LeNet5", build_lenet5),
                        ("LeNet7", build_lenet7)]
    
    # Verify dimensions
    println("\nTesting $name architecture:")
    test_input = rand(Float32, 32, 32, 3, 1)
    for (i, layer) in enumerate(builder())
        test_input = layer(test_input)
        println("Layer $i: ", size(test_input))
    end
    
    # Train and evaluate
    results[name] = train_and_evaluate(builder(), name)
end

# 4. Plot results
plot(collect(keys(results)), collect(values(results)).*100,
    xlabel="Model Variant", 
    ylabel="Test Accuracy (%)",
    title="Filter Size Comparison on CIFAR-10",
    label="Accuracy",
    marker=:circle, linewidth=2)
savefig("filter_size_comparison.png")



""" Question 4. Investigate the learned features of the convolution layers. Using any 3 samples, draw images showing how the image transforms as it passes the convolution layers of the LeNet3
architecture. """

using Flux, MLDatasets, Plots
using Flux: onehotbatch
using MLDatasets: CIFAR10

# 1. Build LeNet3 with correct dimension handling
function build_lenet3()
    Chain(
        # Layer 1: Conv -> ReLU -> Pool
        Conv((3,3), 3 => 6, relu, pad=1),
        x -> maxpool(reshape(x, size(x,1), size(x,2), size(x,3), :), (2,2)),
        
        # Layer 2: Conv -> ReLU -> Pool
        Conv((3,3), 6 => 16, relu, pad=1),
        x -> maxpool(reshape(x, size(x,1), size(x,2), size(x,3), :), (2,2)),
        
        # Flatten
        x -> reshape(x, :, size(x,4)),
        
        # Classifier
        Dense(8*8*16 => 120, relu),
        Dense(120 => 84, relu),
        Dense(84 => 10),
        softmax
    )
end

# Helper function for 4D maxpool
function maxpool(x::Array{Float32,4}, k)
    # Custom implementation to avoid dimension issues
    h,w,c,b = size(x)
    new_h = h ÷ k[1]
    new_w = w ÷ k[2]
    result = zeros(Float32, new_h, new_w, c, b)
    
    for i in 1:new_h, j in 1:new_w, ci in 1:c, bi in 1:b
        patch = x[(i-1)*k[1]+1:i*k[1], (j-1)*k[2]+1:j*k[2], ci, bi]
        result[i,j,ci,bi] = maximum(patch)
    end
    return result
end

# 2. Visualization function
function visualize_features(model, samples)
    # Extract model components
    conv1 = model[1]
    pool1 = model[2]
    conv2 = model[3]
    pool2 = model[4]
    
    plots = []
    for i in 1:size(samples, 4)
        img = samples[:,:,:,i:i]  # Keep as 4D (add batch dim)
        
        # Forward pass
        c1 = conv1(img)
        p1 = pool1(c1)
        c2 = conv2(p1)
        p2 = pool2(c2)
        
        # Visualizations
        p_orig = heatmap(img[:,:,1,1]', title="Original", c=:grays, axis=false)
        p_conv1 = heatmap(hcat([c1[:,:,c,1] for c in 1:6]...), 
                         title="Conv1 Features", c=:viridis, axis=false)
        p_conv2 = heatmap(hcat([c2[:,:,c,1] for c in 1:6]...), 
                         title="Conv2 Features", c=:viridis, axis=false)
        
        push!(plots, plot(p_orig, p_conv1, p_conv2, layout=(1,3), size=(1200,400)))
    end
    return plots
end

# 3. Load data and visualize
model = build_lenet3()
train_x, _ = CIFAR10.traindata(Float32)
samples = train_x[:,:,:,rand(1:50000, 3)] ./ 255f0  # 3 random samples

# Generate and save plots
plots = visualize_features(model, samples)
for (i,p) in enumerate(plots)
    savefig(p, "lenet3_features_sample$i.png")
    display(p)
end