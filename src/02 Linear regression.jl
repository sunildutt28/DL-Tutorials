
# ## Simple linear regression

# `MLJ` essentially serves as a unified path to many existing Julia packages each of which provides their own functionalities and models, with their own conventions.
#
# The simple linear regression demonstrates this.
# Several packages offer it (beyond just using the backslash operator): here we will use `MLJLinearModels` but we could also have used `GLM`, `ScikitLearn` etc.
#
# To load the model from a given package use `@load ModelName pkg=PackageName`

using MLJ
models()

filter(model) = model.is_pure_julia && model.is_supervised && model.prediction_type == :probabilistic
models(filter)
models("XGB")
measures("F1")



# Linear regression

using Pkg    
Pkg.add("MLJLinearModels")

LR = @load LinearRegressor pkg = MLJLinearModels

# Note: in order to be able to load this, you **must** have the relevant package in your environment, if you don't, you can always add it (``using Pkg; Pkg.add("MLJLinearModels")``).
#
# Let's load the _boston_ data set

import RDatasets: dataset
import DataFrames: describe, select, Not, rename!
data = dataset("MASS", "Boston")
println(first(data, 3))

# Let's get a feel for the data

@show describe(data)

# So there's no missing value and most variables are encoded as floating point numbers.
# In MLJ it's important to specify the interpretation of the features (should it be considered as a Continuous feature, as a Count, ...?), see also [this tutorial section](/getting-started/choosing-a-model/#data_and_its_interpretation) on scientific types.
#
# Here we will just interpret the integer features as continuous as we will just use a basic linear regression:

data = coerce(data, autotype(data, :discrete_to_continuous))

# Let's also extract the target variable (`MedV`):

y = data.MedV
X = select(data, Not(:MedV))

mdls = models(matching(X, y))

# Let's declare a simple multivariate linear regression model:

model = LR()

# First let's do a very simple univariate regression, in order to fit it on the data, we need to wrap it in a _machine_ which, in MLJ, is the composition of a model and data to apply the model on:

X_uni = select(X, :LStat) # only a single feature
mach_uni = machine(model, X_uni, y)
fit!(mach_uni)
ŷ = MLJ.predict(mach_uni, X_uni)
round(rsquared(ŷ, y), sigdigits=4)
# You can then retrieve the  fitted parameters using `fitted_params`:

fp = fitted_params(mach_uni)
@show fp.coefs
@show fp.intercept

# You can also visualise this

using Plots

plot(X.LStat, y, seriestype=:scatter, markershape=:circle, legend=false, size=(800, 600), xlabel="LStat")

#  MLJ.predict(mach_uni, Xnew) to predict from a fitted model
Xnew = (LStat=collect(range(extrema(X.LStat)..., length=100)),)
plot!(Xnew.LStat, MLJ.predict(mach_uni, Xnew), linewidth=3, color=:orange)


# The  multivariate linear regression case is very similar

mach = machine(model, X, y)
fit!(mach)

fp = fitted_params(mach)
coefs = fp.coefs
intercept = fp.intercept
for (name, val) in coefs
    println("$(rpad(name, 8)):  $(round(val, sigdigits=3))")
end
println("Intercept: $(round(intercept, sigdigits=3))")

# You can use the `machine` in order to _predict_ values as well and, for instance, compute the root mean squared error:

ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)

# Let's see what the residuals look like

res = ŷ .- y
begin
    plot(res, line=:stem, linewidth=1, marker=:circle, legend=false, size=((800, 600)))
    hline!([0], linewidth=2, color=:red)    # add a horizontal line at x=0
end
mean(y)

# Maybe that a histogram is more appropriate here

histogram(res, normalize=true, size=(800, 600), label="residual")


# ## Interaction and transformation
#
# Let's say we want to also consider an interaction term of `lstat` and `age` taken together.
# To do this, just create a new dataframe with an additional column corresponding to the interaction term:

X2 = hcat(X, X.LStat .* X.Age)

# So here we have a DataFrame with one extra column corresponding to the elementwise products between `:LStat` and `Age`.
# DataFrame gives this a default name (`:x1`) which we can change:

rename!(X2, :x1 => :interaction)

# Ok cool, now let's try the linear regression again

mach = machine(model, X2, y)
fit!(mach)
ŷ = MLJ.predict(mach, X2)
round(rsquared(ŷ, y), sigdigits=4)

# We get slightly better results but nothing spectacular.
#
# Let's consider regressing the target variable on `lstat` and `lstat^2`; again:
using DataFrames
X3 = DataFrame(hcat(X.LStat, X.LStat .^ 2), [:LStat, :LStat2])
mach = machine(model, X3, y)
fit!(mach)
ŷ = MLJ.predict(mach, X3)
round(rsquared(ŷ, y), sigdigits=4)

# fitting y=mx+c to LStat^2 is the same as fitting y=mx2+c to LStat => Polynomial regression

# which again, we can visualise:

Xnew = (LStat=Xnew.LStat, LStat2=Xnew.LStat .^ 2)

plot(X.LStat, y, seriestype=:scatter, markershape=:circle, legend=false, size=(800, 600))
plot!(Xnew.LStat, MLJ.predict(mach, Xnew), linewidth=3, color=:orange)



# TODO HW : Find the best model by feature selection; best model means highest R²


X_uni3 = select(X, :Crim) # only a single feature
mach_uni3 = machine(model, X_uni3, y)
fit!(mach_uni3)
ŷ3 = MLJ.predict(mach_uni3, X_uni3)
round(rsquared(ŷ3, y), sigdigits=4)

X_uni4 = select(X, :Zn) # only a single feature
mach_uni4 = machine(model, X_uni4, y)
fit!(mach_uni4)
ŷ4 = MLJ.predict(mach_uni4, X_uni4)
round(rsquared(ŷ4, y), sigdigits=4)

X_uni5 = select(X, :Indus) # only a single feature
mach_uni5 = machine(model, X_uni5, y)
fit!(mach_uni5)
ŷ5 = MLJ.predict(mach_uni5, X_uni5)
round(rsquared(ŷ5, y), sigdigits=4)

X_uni6 = select(X, :Chas) # only a single feature
mach_uni6 = machine(model, X_uni6, y)
fit!(mach_uni6)
ŷ6 = MLJ.predict(mach_uni6, X_uni6)
round(rsquared(ŷ6, y), sigdigits=4)

X_uni7 = select(X, :NOx) # only a single feature
mach_uni7 = machine(model, X_uni7, y)
fit!(mach_uni7)
ŷ7 = MLJ.predict(mach_uni7, X_uni7)
round(rsquared(ŷ7, y), sigdigits=4)

X_uni8 = select(X, :Rm) # only a single feature
mach_uni8 = machine(model, X_uni8, y)
fit!(mach_uni8)
ŷ8 = MLJ.predict(mach_uni8, X_uni8)
round(rsquared(ŷ8, y), sigdigits=4)

X_uni9 = select(X, :Age) # only a single feature
mach_uni9 = machine(model, X_uni9, y)
fit!(mach_uni9)
ŷ9 = MLJ.predict(mach_uni9, X_uni9)
round(rsquared(ŷ9, y), sigdigits=4)

X_uni10 = select(X, :Dis) # only a single feature
mach_uni10 = machine(model, X_uni10, y) 
fit!(mach_uni10)
ŷ10 = MLJ.predict(mach_uni10, X_uni10)
round(rsquared(ŷ10, y), sigdigits=4)

X_uni11 = select(X, :Rad) # only a single feature
mach_uni11 = machine(model, X_uni11, y) 
fit!(mach_uni11)
ŷ11 = MLJ.predict(mach_uni11, X_uni11)
round(rsquared(ŷ11, y), sigdigits=4)

X_uni12 = select(X, :Tax) # only a single feature
mach_uni12 = machine(model, X_uni12, y)
fit!(mach_uni12)
ŷ12 = MLJ.predict(mach_uni12, X_uni12)
round(rsquared(ŷ12, y), sigdigits=4)

X_uni13 = select(X, :PTRatio) # only a single feature
mach_uni13 = machine(model, X_uni13, y) 
fit!(mach_uni13)
ŷ13 = MLJ.predict(mach_uni13, X_uni13)
round(rsquared(ŷ13, y), sigdigits=4)

X_uni14 = select(X, :Black) # only a single feature
mach_uni14 = machine(model, X_uni14, y) 
fit!(mach_uni14)
ŷ14 = MLJ.predict(mach_uni14, X_uni14)
round(rsquared(ŷ14, y), sigdigits=4)

X_uni15 = select(X, :LStat) # only a single feature
mach_uni15 = machine(model, X_uni15, y)
fit!(mach_uni15)
ŷ15 = MLJ.predict(mach_uni15, X_uni15)
round(rsquared(ŷ15, y), sigdigits=4)
# Now we can compare the R² values of all the models
r_squared_values = [
    ("Crim", round(rsquared(ŷ3, y), sigdigits=4)),
    ("Zn", round(rsquared(ŷ4, y), sigdigits=4)),
    ("Indus", round(rsquared(ŷ5, y), sigdigits=4)),
    ("Chas", round(rsquared(ŷ6, y), sigdigits=4)),
    ("NOx", round(rsquared(ŷ7, y), sigdigits=4)),
    ("Rm", round(rsquared(ŷ8, y), sigdigits=4)),
    ("Age", round(rsquared(ŷ9, y), sigdigits=4)),
    ("Dis", round(rsquared(ŷ10, y), sigdigits=4)),
    ("Rad", round(rsquared(ŷ11, y), sigdigits=4)),
    ("Tax", round(rsquared(ŷ12, y), sigdigits=4)),
    ("PTRatio", round(rsquared(ŷ13, y), sigdigits=4)),
    ("B", round(rsquared(ŷ14, y), sigdigits=4)),
    ("LStat", round(rsquared(ŷ15, y), sigdigits=4))
]
# find highest rsquared value in vector 

X = select(data, Not(:MedV,:LStat)) # remove LStat to avoid multicollinearity)
mach = machine(model, X, y)
fit!(mach)
ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)


X = select(data, Not(:MedV,:Black))
mach = machine(model, X, y)
fit!(mach)
ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)

X = select(data, Not(:MedV,:PTRatio))
mach = machine(model, X, y) 
fit!(mach)
ŷ = MLJ.predict(mach, X)    
round(rsquared(ŷ, y), sigdigits=4)


X = select(data, Not(:MedV,:Tax))
mach = machine(model, X, y) 
fit!(mach)
ŷ = MLJ.predict(mach, X)    
round(rsquared(ŷ, y), sigdigits=4)


X = select(data, Not(:MedV,:Rad))
mach = machine(model, X, y) 
fit!(mach)
ŷ = MLJ.predict(mach, X)    
round(rsquared(ŷ, y), sigdigits=4)

X = select(data, Not(:MedV,:Dis))
mach = machine(model, X, y) 
fit!(mach)
ŷ = MLJ.predict(mach, X)    
round(rsquared(ŷ, y), sigdigits=4)

X = select(data, Not(:MedV,:Age))
mach = machine(model, X, y)
fit!(mach)
ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)

X = select(data, Not(:MedV,:Rm))
mach = machine(model, X, y)
fit!(mach)
ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)      

X = select(data, Not(:MedV,:NOx))
mach = machine(model, X, y) 
fit!(mach)
ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)

X = select(data, Not(:MedV,:Chas))
mach = machine(model, X, y)
fit!(mach)
ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)

X = select(data, Not(:MedV,:Indus))
mach = machine(model, X, y)
fit!(mach)
ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)  

X = select(data, Not(:MedV,:Zn))
mach = machine(model, X, y) 
fit!(mach)
ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)

X = select(data, Not(:MedV,:Crim))
mach = machine(model, X, y) 
fit!(mach)
ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)

# if we drop induc or age we have highest r_squared_values 74.06

X = select(data, Not(:MedV,:Indus, :Age)) # remove Indus and Age to avoid multicollinearity
mach = machine(model, X, y) 
fit!(mach)
ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)
