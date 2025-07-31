import Pkg; Pkg.add("MLDataUtils")
using Flux, DataFrames, CSV, Statistics, MLDataUtils, CategoricalArrays, JLD2

# Load the dataset
df = CSV.read("Group A Dataset.csv", DataFrame)

# Preprocessing
# Remove specified columns
select!(df, Not([:fnlwgt, :education]))

# Convert categorical variables to numerical
categorical_cols = [:workclass, :marital_status, :occupation, :relationship, 
                    :race, :sex, :native_country, :label]

for col in categorical_cols
    df[!, col] = categorical(df[!, col])
    levels = unique(df[!, col])
    df[!, col] = levelcode.(df[!, col])
end

# Split into features and labels
X = Matrix(select(df, Not(:label)))
y = df.label .== 2  # Assuming >50K is encoded as 2

# Split into train and test sets (80-20 split)
train_idx, test_idx = stratifiedobs((X, y), p=0.8)
X_train, y_train = X[:, train_idx], y[train_idx]
X_test, y_test = X[:, test_idx], y[test_idx]

# Normalize features
function normalize_data!(X)
    μ = mean(X, dims=2)
    σ = std(X, dims=2)
    X .= (X .- μ) ./ (σ .+ 1e-6)
    return X, μ, σ
end

X_train, μ, σ = normalize_data!(X_train)
X_test = (X_test .- μ) ./ (σ .+ 1e-6)

# Define model architecture with parameter constraint
function build_model(input_size)
    # Calculate layer sizes to stay under 1000 parameters
    hidden1 = min(32, floor(Int, (1000 - 2) / (input_size + 2)))
    hidden2 = min(16, floor(Int, (1000 - hidden1 - 2) / (hidden1 + 2)))
    
    Chain(
        Dense(input_size, hidden1, relu),
        Dense(hidden1, hidden2, relu),
        Dense(hidden2, 2),
        softmax
    )
end

model = build_model(size(X_train, 1))

# Verify parameter count
println("Total parameters: ", sum(length, Flux.params(model)))  # Should be ≤ 1000

# Define loss function
loss(x, y) = Flux.logitcrossentropy(model(x), y)

# Prepare data in minibatches
train_data = Flux.DataLoader((X_train, Flux.onehotbatch(y_train, [false, true])), batchsize=128, shuffle=true)
test_data = Flux.DataLoader((X_test, Flux.onehotbatch(y_test, [false, true])), batchsize=128)

# CORRECTED: Proper AdamW implementation
opt = Flux.AdamW(0.001, (0.9, 0.999), 0.001)  # (lr, β, weight_decay)

# Training function
function train_model!(model, train_data, test_data, opt, epochs)
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model), train_data, opt)
        
        # Calculate training accuracy
        train_acc = mean(Flux.onecold(model(X_train)) .- 1 .== y_train)
        
        # Calculate test accuracy
        test_acc = mean(Flux.onecold(model(X_test)) .- 1 .== y_test)
        
        # Calculate balanced accuracy
        y_pred_train = Flux.onecold(model(X_train)) .- 1
        y_pred_test = Flux.onecold(model(X_test)) .- 1
        
        # True positives, true negatives
        tp_train = sum((y_pred_train .== 1) .& (y_train .== 1))
        tn_train = sum((y_pred_train .== 0) .& (y_train .== 0))
        tp_test = sum((y_pred_test .== 1) .& (y_test .== 1))
        tn_test = sum((y_pred_test .== 0) .& (y_test .== 0))
        
        # Sensitivity and specificity
        sens_train = tp_train / sum(y_train .== 1)
        spec_train = tn_train / sum(y_train .== 0)
        sens_test = tp_test / sum(y_test .== 1)
        spec_test = tn_test / sum(y_test .== 0)
        
        bal_acc_train = (sens_train + spec_train) / 2
        bal_acc_test = (sens_test + spec_test) / 2
        
        println("Epoch $epoch: Train Acc = $(round(train_acc, digits=4)), Test Acc = $(round(test_acc, digits=4)), Balanced Test Acc = $(round(bal_acc_test, digits=4))")
    end
    return bal_acc_test
end

# Train the model
final_bal_acc = train_model!(model, train_data, test_data, opt, 100)

println("\nFinal Balanced Accuracy: $(round(final_bal_acc, digits=4))")

# Save the model
JLD2.@save "income_model.jld2" model μ σ