using Lux, Optimisers, Zygote
using MLUtils, MLDatasets, OneHotArrays
using Statistics, Random
using Plots

# Set seed for reproducibility
Random.seed!(42)

# Fixed data loader
function getdata(batchsize=128)
    xtrain, ytrain = FashionMNIST(split=:train)[:]
    xtest, ytest = FashionMNIST(split=:test)[:]
    
    xtrain = Float32.(reshape(xtrain, 28*28, :)) ./ 255.0f0
    xtest = Float32.(reshape(xtest, 28*28, :)) ./ 255.0f0
    
    ytrain = onehotbatch(ytrain, 0:9)
    ytest = onehotbatch(ytest, 0:9)
    
    return (
        DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true),
        DataLoader((xtest, ytest), batchsize=batchsize)
    )
end

# Model definition
function create_model(hidden_size)
    Chain(
        Dense(28*28 => hidden_size, relu),
        Dense(hidden_size => 10),
    )
end

# Fixed training function
function train_model(hidden_size; epochs=10, lr=0.001)
    train_loader, test_loader = getdata()
    model = create_model(hidden_size)
    ps, st = Lux.setup(Random.default_rng(), model)
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, ps)
    
    for epoch in 1:epochs
        # Training phase
        for (x, y) in train_loader
            loss, back = Zygote.pullback(ps) do p
                ŷ, _ = model(x, p, st)
                -mean(sum(y .* logsoftmax(ŷ); dims=1))
            end
            gs = back(1f0)[1]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
        end
        
        # Testing phase - FIXED VARIABLE SCOPE
        test_acc = 0.0
        count = 0
        for (x, y) in test_loader
            ŷ, _ = model(x, ps, st)
            test_acc += mean(onecold(ŷ) .== onecold(y)) * size(y, 2)
            count += size(y, 2)
        end
        test_acc /= count
        println("Size $hidden_size | Epoch $epoch | Acc: $(round(test_acc*100, digits=2))%")
    end
    
    # Final test accuracy
    final_acc = 0.0
    final_count = 0
    for (x, y) in test_loader
        ŷ, _ = model(x, ps, st)
        final_acc += mean(onecold(ŷ) .== onecold(y)) * size(y, 2)
        final_count += size(y, 2)
    end
    return final_acc / final_count
end

# Run experiment
hidden_sizes = [10, 20, 40, 50, 100, 300]
accuracies = Float64[]

@time for size in hidden_sizes
    acc = train_model(size)
    push!(accuracies, acc)
    println("Completed size $size with accuracy $(round(acc*100, digits=2))%")
end

# Plot results
plot(hidden_sizes, accuracies, xlabel="Hidden Size", ylabel="Accuracy",
     title="FashionMNIST MLP Performance", legend=false, marker=:circle)
savefig("fashionmnist_results.png")