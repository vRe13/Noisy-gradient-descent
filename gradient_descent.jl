using LinearAlgebra
using Statistics
using Plots
using StaticArrays
using Random

# Simulation parameters
N = 50 # number of particles
d = 2 # dimension
T = 100000 # number of time steps
dt = 0.0001 # time step
sigma = 0.2 # noise intensity
eps = 0.5 # penalty term for the neural network #1.2

# Generate random data with covariance matrix diag(1,4)
# X = [randn(1,100); 2*randn(1,100)] # 100 test points with variance 1 and 4

# Compute covariance matrix of the data
#cov_X = cov(X')  # Note: transpose needed for proper covariance
cov_X = [2.5 -1.5; -1.5 2.5]

mutable struct weights
    W_0::Matrix{Float64}
    W_1::Matrix{Float64}
    mean_0::Vector{Float64}
    mean_1::Vector{Float64}
end

function grad_V0(cov::Matrix{Float64}, mean_0::Vector{Float64}, mean_1::Vector{Float64})
    # Compute the gradient based on the particles
    return - 2 / length(mean_0) * (cov * (mean_1 - norm(mean_1)^2 * mean_0))
end

function grad_V1(cov::Matrix{Float64}, mean_0::Vector{Float64}, mean_1::Vector{Float64})
    cov_mean_0 = cov * mean_0
    # Compute the gradient based on the particles
    return - 2 / length(mean_1) * ((cov_mean_0 - dot(cov_mean_0, mean_0) * mean_1))
end

# Overdamped Langevin dynamics
function langevin_step!(system::weights, cov::Matrix{Float64}, dt::Float64, sigma::Float64)
    # Unpack weights
    W_0 = system.W_0
    W_1 = system.W_1
    mean_0 = system.mean_0
    mean_1 = system.mean_1
    
    # Compute the gradient
    grad_0 = grad_V0(cov, mean_0, mean_1)
    grad_1 = grad_V1(cov, mean_0, mean_1)
    
    # Update weights using Euler scheme
    W_0 .-= dt * grad_0 * ones(1, size(W_0, 2)) .+ sigma * sqrt(2 * dt) * randn(size(W_0)) .+ 2. * eps * W_0 * dt
    W_1 .-= dt * grad_1 * ones(1, size(W_1, 2)) .+ sigma * sqrt(2 * dt) * randn(size(W_1)) .+ 2. * eps * W_1 * dt
    
    # Compute the new means
    system.mean_0 = vec(mean(W_0, dims=2))
    system.mean_1 = vec(mean(W_1, dims=2))
end

# Initialize particles
particles_0 = sigma / sqrt(2 * eps) * randn(d, N) .- 0.61237 # N particles in d dimensions, shifted by -0.7
particles_1 = sigma / sqrt(2 * eps) * randn(d, N) .- 0.61237 # N particles in d dimensions, shifted by -0.7
mean_0 = vec(mean(particles_0, dims=2))
mean_1 = vec(mean(particles_1, dims=2))
system = weights(particles_0, particles_1, mean_0, mean_1)

# Run simulation
for t in 1:T
    langevin_step!(system, cov_X, dt, sigma)
    
    # Optional: visualize every 100 steps
    if t % 100 == 0
        scatter(system.W_0[1, :], system.W_0[2, :],
                title="Step $t", xlim=(-5,5), ylim=(-5,5),
                markersize=2, label="W_0", alpha=0.75)
        scatter!(system.W_1[1, :], system.W_1[2, :],
                markersize=2, label="W_1", alpha=0.75)
        display(current())
    end
end

# Print final means
println("Final mean_0: ", system.mean_0)
println("Final mean_1: ", system.mean_1)
# Plot final norms 
println("Final squared norm of mean_0: ", norm(system.mean_0)^2)
println("Final squared norm of mean_1: ", norm(system.mean_1)^2)
println("Final theoretical squared norm: ", (4 - eps)/4)