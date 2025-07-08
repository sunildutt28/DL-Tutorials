using Lux, Optimisers, Zygote
using MLUtils, MLDatasets, OneHotArrays
using Statistics, Random
using Plots

# Set seed for reproducibility
Random.seed!(42)

# Q1. Implement a one hidden layer MLP, and vary the size of the hidden layer (10, 20, 40, 50, 100,
#  300) and train for 10 Epochs on FashionMNIST and store the final test accuracy. Then plot
#  the accuracy as a function of the hidden layer size.


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


# Q2. Use the same network with fixed hidden layer size of 30 to estimate the impact of random initialisation. Run the network 10 times with different weight initialisation. Compute standard
#   deviation and mean. Visualize the datapoints in a plot to make the fluctuations of the final
#   test accuracy visible.

# Modified training function to accept manual RNG
function train_model(rng::AbstractRNG, hidden_size=30; epochs=10, lr=0.001)
    train_loader, test_loader = getdata()
    model = Chain(
        Dense(28*28 => hidden_size, relu),
        Dense(hidden_size => 10)
    )
    
    # Use provided RNG for initialization
    ps, st = Lux.setup(rng, model)
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, ps)
    
    for epoch in 1:epochs
        for (x, y) in train_loader
            loss, back = Zygote.pullback(ps) do p
                ŷ, _ = model(x, p, st)
                -mean(sum(y .* logsoftmax(ŷ); dims=1))
            end
            gs = back(1f0)[1]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
        end
    end
    
    # Calculate final test accuracy
    acc = 0.0
    count = 0
    for (x, y) in test_loader
        ŷ, _ = model(x, ps, st)
        acc += mean(onecold(ŷ) .== onecold(y)) * size(y, 2)
        count += size(y, 2)
    end
    return acc / count
end

# Run 10 experiments with different initializations
const hidden_size = 30
const num_runs = 10
accuracies = Float64[]

for i in 1:num_runs
    # Create new RNG for each run
    rng = Random.MersenneTwister(rand(1:10000))
    acc = train_model(rng, hidden_size)
    push!(accuracies, acc)
    println("Run $i: Accuracy = $(round(acc*100, digits=2))%")
end

# Compute statistics
μ = mean(accuracies)
σ = std(accuracies)

println("\nResults after $num_runs runs:")
println("Mean accuracy: $(round(μ*100, digits=2))%")
println("Standard deviation: $(round(σ*100, digits=2))%")

# Visualization
scatter(1:num_runs, accuracies, 
       xlabel="Run Number", ylabel="Test Accuracy",
       title="Impact of Random Initialization (Hidden Size = $hidden_size)",
       label="Individual Runs", ylim=(0.7, 0.9))
hline!([μ], label="Mean Accuracy", linewidth=2)
hline!([μ - σ, μ + σ], label="±1 Std Dev", linestyle=:dash, linewidth=1)
annotate!((num_runs+0.5, μ, text("μ = $(round(μ*100, digits=2))%", 10)))
annotate!((num_runs+0.5, μ-σ, text("σ = $(round(σ*100, digits=2))%", 10)))

savefig("initialization_impact.png")


# Q3. Train the model with a batch size of 32 for 25 epochs. Use a decaying learning rate schedule
#   of your choice.


# Learning rate scheduler (exponential decay)
struct ExpDecaySchedule
    initial_lr::Float32
    decay_rate::Float32
end

function (sched::ExpDecaySchedule)(epoch)
    sched.initial_lr * exp(-sched.decay_rate * (epoch - 1))
end

# Model definition
function create_model(hidden_size=30)
    Chain(
        Dense(28*28 => hidden_size, relu),
        Dense(hidden_size => 10),
    )
end

# Training function with LR scheduling
function train_model(; hidden_size=30, epochs=25)
    train_loader, test_loader = getdata()
    model = create_model(hidden_size)
    ps, st = Lux.setup(Random.default_rng(), model)
    
    # Initialize optimizer with LR schedule
    initial_lr = 0.01f0
    lr_schedule = ExpDecaySchedule(initial_lr, 0.1f0)  # Decays by 10% each epoch
    opt = Optimisers.Adam(initial_lr)
    opt_state = Optimisers.setup(opt, ps)
    
    # Track training progress
    train_losses = Float32[]
    test_accuracies = Float32[]
    
    for epoch in 1:epochs
        # Update learning rate
        current_lr = lr_schedule(epoch)
        Optimisers.adjust!(opt_state, current_lr)
        
        # Training phase
        epoch_loss = 0.0f0
        for (x, y) in train_loader
            loss, back = Zygote.pullback(ps) do p
                ŷ, _ = model(x, p, st)
                -mean(sum(y .* logsoftmax(ŷ); dims=1))
            end
            gs = back(1f0)[1]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
            epoch_loss += loss
        end
        push!(train_losses, epoch_loss / length(train_loader))
        
        # Testing phase
        acc = 0.0f0
        for (x, y) in test_loader
            ŷ, _ = model(x, ps, st)
            acc += mean(onecold(ŷ) .== onecold(y))
        end
        acc /= length(test_loader)
        push!(test_accuracies, acc)
        
        println("Epoch $epoch | LR: $(round(current_lr, digits=5)) | " *
                "Loss: $(round(train_losses[end], digits=4)) | " *
                "Acc: $(round(acc*100, digits=2))%")
    end
    
    return train_losses, test_accuracies, ps
end

# Run training
train_losses, test_accuracies, ps = train_model(epochs=25)

# Plot results
p1 = plot(train_losses, xlabel="Epoch", ylabel="Loss", label="Training Loss", title="Training Curve")
p2 = plot(test_accuracies, xlabel="Epoch", ylabel="Accuracy", label="Test Accuracy", ylim=(0,1))
plot(p1, p2, layout=(2,1), size=(800,600))
savefig("training_results.png")

# Final evaluation
final_acc = mean(test_accuracies[end-4:end])  # Average last 5 epochs
println("\nFinal Test Accuracy: $(round(final_acc*100, digits=2))%")



# Q4. Optimise the batch size and the learning rate schedule via a small grid search.

using Lux, Optimisers, Zygote
using MLUtils, MLDatasets, OneHotArrays
using Statistics, Random
using Plots
using ProgressMeter

# First define all necessary types and functions
abstract type LRSchedule end


struct ExpDecayScheduleGrid <: LRSchedule
    initial_lr::Float32
    decay_rate::Float32
end

function (sched::ExpDecayScheduleGrid)(epoch)
    sched.initial_lr * exp(-sched.decay_rate * (epoch - 1))
end

struct StepDecaySchedule <: LRSchedule
    initial_lr::Float32
    drop_every::Int
    drop_factor::Float32
end

function (sched::StepDecaySchedule)(epoch)
    sched.initial_lr * (sched.drop_factor)^floor((epoch-1)/sched.drop_every)
end

function create_model(hidden_size=30)
    Chain(
        Dense(28*28 => hidden_size, relu),
        Dense(hidden_size => 10),
    )
end

function getdata(batchsize)
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

# Define the training function with concrete types
function train_model(batchsize::Int, lr_schedule::LRSchedule; hidden_size=30, epochs=25)
    train_loader, test_loader = getdata(batchsize)
    model = create_model(hidden_size)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    
    opt = Optimisers.Adam(lr_schedule(1))
    opt_state = Optimisers.setup(opt, ps)
    
    for epoch in 1:epochs
        current_lr = lr_schedule(epoch)
        Optimisers.adjust!(opt_state, current_lr)
        
        for (x, y) in train_loader
            loss, back = Zygote.pullback(ps) do p
                ŷ, _ = model(x, p, st)
                -mean(sum(y .* logsoftmax(ŷ); dims=1))
            end
            gs = back(1f0)[1]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
        end
    end
    
    # Final accuracy calculation
    acc = 0.0f0
    count = 0
    for (x, y) in test_loader
        ŷ, _ = model(x, ps, st)
        acc += mean(onecold(ŷ) .== onecold(y)) * size(y, 2)
        count += size(y, 2)
    end
    return acc / count
end

#  Now run the grid search with all dependencies defined
Random.seed!(42)

batch_sizes = [16, 32, 64]
lr_schedules = [
    ExpDecayScheduleGrid(0.01f0, 0.05f0),
    ExpDecayScheduleGrid(0.01f0, 0.1f0),
    StepDecaySchedule(0.01f0, 5, 0.5f0)
]

results = Dict()
progress = Progress(length(batch_sizes)*length(lr_schedules), desc="Grid searching:")

for batchsize in batch_sizes
    for lr_schedule in lr_schedules
        acc = train_model(batchsize, lr_schedule)  # This should now work
        
         if lr_schedule isa ExpDecayScheduleGrid
            schedule_name = "Exp(λ=$(lr_schedule.decay_rate))"
        elseif lr_schedule isa StepDecaySchedule
            schedule_name = "Step($(lr_schedule.drop_every))"
        else
            schedule_name = "Unknown"
        end

        results[(batchsize, schedule_name)] = acc
        
        next!(progress; showvalues=[
            (:BatchSize, batchsize),
            (:Schedule, schedule_name),
            (:Accuracy, round(acc*100, digits=2))
        ])
    end
end

#  Results analysis
best = argmax(results)
println("\nBest config: Batch $(best[1]), $(best[2]) → $(round(results[best]*100, digits=2))%")


# Prepare axes
schedules = ["Exp(λ=0.05)", "Exp(λ=0.1)", "Step(5)"]
batches = [16, 32, 64]

# Build accuracy matrix
acc_matrix = [results[(b, s)] for b in batches, s in schedules]


# Plot heatmap for visualization
hm = heatmap(
    schedules, batches, acc_matrix,
    xlabel="Schedule", ylabel="Batch Size", title="Accuracy Heatmap"
)
savefig("grid_results.png")

# Q5. Train the model with the best parameters found in the grid search and report the final test accuracy.
println("\nTraining with best grid search parameters:")

# Find the actual schedule object for the best config
function find_schedule(schedule_name, lr_schedules)
    for sched in lr_schedules
        if sched isa ExpDecayScheduleGrid && schedule_name == "Exp(λ=$(sched.decay_rate))"
            return sched
        elseif sched isa StepDecaySchedule && schedule_name == "Step($(sched.drop_every))"
            return sched
        end
    end
    error("Schedule not found")
end

best_batch = best[1]
best_sched_name = best[2]
best_sched = find_schedule(best_sched_name, lr_schedules)

println("\nTraining with best grid search parameters:")
best_acc = train_model(best_batch, best_sched)
println("Final test accuracy with best config: $(round(best_acc*100, digits=2))%")

# Compare to Q3
println("\nQ3 Final Test Accuracy: $(round(final_acc*100, digits=2))%")
println("Grid Search Best Test Accuracy: $(round(best_acc*100, digits=2))%")

#Output = Final test accuracy with best config: 85.51%


#Output = Q3 Final Test Accuracy: 85.19%

#Output = Grid Search Best Test Accuracy: 85.51%

# Improved the results by 0.32% with grid search.
# This shows the effectiveness of hyperparameter tuning in improving model performance.
