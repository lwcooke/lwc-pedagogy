"""
++ RocketUtils Module++

Utility functions for rocket-trajectory simulations.
"""
module RocketUtils

using LinearAlgebra, OrdinaryDiffEq

export Body, solve_rocket_ODE, obj_circle, solve_mb_ODE

"""
    Body(m<:Float64, r<:Vector{Float64}, s<:Float64, label<:String, v=[0. ; 0.])

Define a new massive object at a point in space.

# Arguments:
- `m::Float64` : Mass of the object.
- `r::Vector{Float64}` : Position vector of the object.
- `s::Float64` : Radius of the object assumed circular.
- `label::String` : Name of the object.
- `v::Vector{Float64}` : Velocity of object, by default set to [0. ; 0.].
"""
Base.@kwdef struct Body
    m::Float64
    r::Vector{Float64}
    s::Float64
    label::String
    v::Vector{Float64} = [0. ; 0.]
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


"""
    gravity_mb(r1, r2, m1, m2; G)

Compute the gravitational acceleration vector target (r1, m1) and another object (r2, m2)

# Arguments:
- `r1` : Position vector of the target
- `m1` : Mass of the target.
- `r2` : Position vector of the other object.
- `m2` : Mass of the other object.
- `G` : Gravitational constant
"""
function gravity_mb(r1, m1, r2, m2; G)
    r21 = (r2 - r1) / norm(r1 - r2)  # Vector between objects
    return G * m1 * m2 * r21 / norm(r1 - r2)^2  # Vectorized gravity
end


"""
    mb_ODE(r, p, t)

Many-body ODE problem, integrating r & ṙ for many objects with purely Newtonian gravity.

# Arguments
- `r` : Vector of coordinate. Must be alternating [r1... ; v1... ; r2... ; v2...]
    with ri, vi position and velocity of ith object.
- `p` : Parameters which contains, in this order, (mass_list, length(ri), obj_size_list, G)
- `t` : Time, for integrator.
"""
function mb_ODE(r, p, t)
    ms, dims, G = p[1], p[2], p[4]  # Masses, dimensions, G

    # Seperate into r and ṙ
    rs = [[r[i], r[i+1]] for i in 1:Int(2 * dims):length(r)]
    vs = [[r[i], r[i+1]] for i in (dims + 1):Int(2 * dims):length(r)]

    # r̈
    ac = [zeros(Float64, dims) for i in 1:length(rs)]
    for i in 1:length(rs), j in 1:length(rs)
        if i != j
            ac[i] += gravity_mb(rs[i], ms[i], rs[j], ms[j]; G)
        end
    end

    # Arrange output vector
    dr = Float64[]
    for i in 1:length(rs)
        push!(dr, vs[i]...)
        push!(dr, ac[i]...)
    end

    return dr
end


"""
    check_collision(r, integrator)

Function for checking if any objects in r are within the radii of one another, flagging a collision.

# Arguments
- `r` : Vector of coordinate. Must be alternating [r1... ; v1... ; r2... ; v2...]
    with ri, vi position and velocity of ith object.
- `integrator` : ODE solver, see OrdinaryDiffEq.jl
"""
function check_collision(r, integrator)
    dims = integrator.p[2]
    sizes = integrator.p[3]
    rs = [[r[i], r[i+1]] for i in 1:Int(2 * dims):length(r)]
    vs = [[r[i], r[i+1]] for i in (dims + 1):Int(2 * dims):length(r)]

    flag, inds = false, Vector{Int}[]
    for i in 1:length(rs), j in i:length(rs)
        if i != j && norm(rs[i] - rs[j]) <= sizes[i] + sizes[j]
            flag = true
            push!(inds, [i ; j])
        end
    end
    return flag, inds, rs, vs
end


"""
    collision_affect!(integrator)

Computes effects of elastic collision between objects. Calls check_collision for affected bodies.

# Arguments
- `integrator` : ODE solver, see OrdinaryDiffEq.jl
"""
function collision_affect!(integrator)
    let r = integrator.u, ms = integrator.p[1]
        fl, inds, rs, vs = check_collision(r, integrator)

        for (i, j) in inds
            v1, v2 = vs[i], vs[j]
            m1, m2 = ms[i], ms[j]
            vf1 = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
            vf2 = (2 * m1 * v1 + (m2 - m1) * v2) / (m1 + m2)
            vs[i], vs[j] = vf1, vf2
        end

        # Arrange output vector
        dr = Float64[]
        for i in 1:length(rs)
            push!(dr, rs[i]...)
            push!(dr, vs[i]...)
        end
        integrator.u = dr
    end
end


"""
    solve_mb_ODE(bods::Vector{Body}, tspan; G)

Solve the many body problem for list of objects, `bods` providing initial conditions.

# Arguments:
- `bods::Vector{Body}` : List of Body structs providing initial conditions and attributes of each body.
- `tspan` : Time-span for simulation.
- `G` : Gravitational constant.
"""
function solve_mb_ODE(bods::Vector{Body}, tspan; G)

    # Unpack Body structs
    r0, mass, size = Float64[], Float64[], Float64[]
    for i in 1:lastindex(bods)
        push!(r0, bods[i].r...)
        push!(r0, bods[i].v...)
        push!(mass, bods[i].m)
        push!(size, bods[i].s)
    end

    # Elastic collision callback
    condition(r, t, integrator) = check_collision(r, integrator)[1]
    cb = DiscreteCallback(condition, collision_affect!)

    prob = ODEProblem(mb_ODE, r0, (0.0, tspan), (mass, length(bods[1].r), size, G))
    return solve(prob, Tsit5(), callback=cb)
end


end  # Module
