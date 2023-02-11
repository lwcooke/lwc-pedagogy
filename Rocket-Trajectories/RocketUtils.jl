"""
++ RocketUtils Module++

Utility functions for rocket-trajectory simulations.
"""
module RocketUtils

using LinearAlgebra, OrdinaryDiffEq

export Body, solve_rocket_ODE, obj_circle

const G = 1e-4  # Gravitational constant

"""
    Body{m::Float64, r::Vector{Float64}, s::Float64, label::String}

Define a new massive object at a point in space.

# Arguments:
- `m::Float64` : Mass of the object.
- `r::Vector{Float64}` : Position vector of the object.
- `s::Float64` : Radius of the object.
- `label::String` : Name of the object.
"""
struct Body{m<:Float64,r<:Vector{Float64},s<:Float64,label<:String}
    m::m
    r::r
    s::s
    label::label
end


"""
    gravity(r, m, obj::Body)

Compute the gravitational acceleration vector between target mass (m @ r) and object.

# Arguments:
- `r` : Position vector of the target
- `m` : Mass of the target.
- `obj::Body` : Object to compute gravitational acceleration from
"""
function gravity(r, m, obj::Body)
    r21 = (obj.r - r) / norm(r - obj.r)  # Vector between objects
    return G * m * obj.m * r21 / norm(r - obj.r)^2  # Vectorized gravity
end


"""
    rocket_ODE(r, p, t)

ODE step function for target mass interacting gravitationally with several fixed objects

# Arguments:
- `r` : Vector of positions and velocities, in the form [r... ; ṙ...].
- `p` : Parameters of form [target_mass, length(r)/2, [objects...]].
- `t` : Time, for integrator.
"""
function rocket_ODE(r, p, t)
    m, halfdim, bods = p

    # Acceleration
    a = zeros(halfdim)
    for b in bods
        a += gravity(r[1:halfdim], m, b)
    end

    return [r[halfdim+1:2*halfdim]; a]
end


"""
    solve_rocket_ODE(m, r0, v0, tspan, objs)

Solve the rocket_ODE problem for a target mass with initial conditions.

# Arguments:
- `m` : Target mass.
- `r0` : Target's initial position vector.
- `v0` : Target's initial velocity vector.
- `tspan` : Time-span for simulation.
- `objs` : List of gravitational objects.
"""
function solve_rocket_ODE(m, r0, v0, tspan, objs)
    # Set collision condition
    condition(r, t, integrator) = Bool(sum(
        [norm(ob.r - r[1:length(r0)]) < ob.s for ob in objs]
    ))  # If target comes within an object radius
    affect!(integrator) = terminate!(integrator)  # Terminate integration
    cb = DiscreteCallback(condition, affect!)

    prob = ODEProblem(rocket_ODE, [r0...; v0...], (0.0, tspan), (m, length(r0), objs))
    return solve(prob, Tsit5(), callback=cb)
end


"""
    obj_circle(ob::Body)

Return points in a circle around Body based on its position and size
"""
function obj_circle(ob::Body)
    θ = range(0, 2π; length=500)
    ob.r[1] .+ ob.s * sin.(θ), ob.r[2] .+ ob.s * cos.(θ)
end


end  # Module
