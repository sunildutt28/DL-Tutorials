using Lux, Optimisers, Zygote
using MLUtils, MLDatasets, OneHotArrays
using Statistics, Random
using Plots

# Set seed for reproducibility
Random.seed!(42)

# Custom crossentropy function
function crossentropy(y_pred, y_true)
    logp = logsoftmax(y_pred)
    return -mean(sum(y_true .* logp; dims=1))
end

# Load FashionMNIST dataset
function getdata(batchsize=128)
    # Load training data
    xtrain, ytrain = FashionMNIST(split=:train)[:]
    xtrain = Float32.(reshape(xtrain, 28*28, :)) ./ 255.0
    ytrain = onehotbatch(ytrain, 0:9)
    
    # Load test data
    xtest, ytest = FashionMNIST(split=:test)[:]
    xtest = Float32.(reshape(xtest, 28*28, :)) ./ 255.0
    ytest = onehotbatch(ytest, 0:9)
    
    # Create data loaders
    train_loader = DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=batchsize)
    
    return train_loader, test_loader
end

# Define model architecture
function create_model(hidden_size)
    Chain(
        Dense(28*28, hidden_size, relu),
        Dense(hidden_size, 10),
    )
end

# Training function
function train_model(hidden_size; epochs=10, batchsize=128)
    # Get data
    train_loader, test_loader = getdata(batchsize)
    
    # Create model
    model = create_model(hidden_size)
    ps, st = Lux.setup(Random.default_rng(), model)
    
    # Loss function
    function loss(x, y)
        y_pred, _ = model(x, ps, st)
        return crossentropy(y_pred, y)
    end
    
    # Optimizer
    opt = Optimisers.Adam(0.001)
    opt_state = Optimisers.setup(opt, ps)
    
    # Training loop
    for epoch in 1:epochs
        for (x, y) in train_loader
            gs = gradient(p -> loss(x, y), ps)[1]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
        end
        println("Epoch $epoch completed for hidden size $hidden_size")
    end
    
    # Compute test accuracy
    accuracy = 0.0
    for (x, y) in test_loader
        y_pred, _ = model(x, ps, st)
        accuracy += mean(onecold(y_pred) .== onecold(y))
    end
    accuracy /= length(test_loader)
    
    return accuracy
end

# Hidden sizes to test
hidden_sizes = [10, 20, 40, 50, 100, 300]
accuracies = Float64[]

# Train models with different hidden sizes
for size in hidden_sizes
    println("\nTraining model with hidden size: $size")
    acc = train_model(size)
    push!(accuracies, acc)
    println("Test accuracy: $(round(acc*100, digits=2))%")
end

# Plot results
plot(hidden_sizes, accuracies, xlabel="Hidden Layer Size", ylabel="Test Accuracy", 
     label="Accuracy", marker=:circle, title="MLP Performance on FashionMNIST")
savefig("mlp_hidden_size_vs_accuracy.png")