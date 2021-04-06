import Pkg; Pkg.activate(joinpath(@__DIR__)); 
Pkg.instantiate();


using ConstrainedDynamics
using ConstrainedDynamicsVis
using ConstrainedControl
using StaticArrays
using Rotations

const RS = Rotations
const CD = ConstrainedDynamics
const CDV = ConstrainedDynamicsVis
const CC = ConstrainedControl

""" helper functions for simple two link system """
function fullargsinds(i)
    # x, v, q, ω
    return 13*(i-1) .+ (1:3), 
            13*(i-1) .+ (4:6), 
            13*(i-1) .+ (7:10), 
            13*(i-1) .+ (11:13)
end
function getStates(mech, sol=true)
    nb = length(mech.bodies)
    z = zeros(13*nb)
    for (i, body) in enumerate(mech.bodies)        
        xinds, vinds, qinds, ωinds = fullargsinds(i)
        f = sol ? CD.fullargssol : CD.fullargsc
        z[xinds],z[vinds],q,z[ωinds] = f(body.state)
        z[qinds] = RS.params(q)
    end
    return z
end

# state config x v q w 
function generate_config(mech, body_pose, rotations)
    pin = zeros(3)
    pin[1] = body_pose[1]
    pin[2] = body_pose[2]
    pin[3] = body_pose[3]
    prev_q = UnitQuaternion(RotZ(body_pose[4]))
    state = [pin;zeros(3);RS.params(prev_q);zeros(3)]
    pin = pin+prev_q * [mech.bodies[1].shape.xyz[1]/2,0,0]
    for i = 1:length(rotations)
        r = UnitQuaternion(rotations[i])
        link_q = prev_q * r
        delta = link_q * [arm_width/2,0,0]
        link_x = pin+delta
        state = [state; link_x;zeros(3);RS.params(link_q);zeros(3)]

        prev_q = link_q
        pin += 2*delta
    end


    return state
end

# body pose is x y z thetaz, θ only have z rotation
function generate_config(mech, body_pose::Vector{<:Number}, θ::Vector{<:Number})
    rotations = []
    # joints are arranged orthogonally
    for i=1:length(θ)
        push!(rotations, UnitQuaternion(RotZ(θ[i])))
    end
    return generate_config(mech, body_pose, rotations)
end

function setStates!(mech, z)
    for (i, body) in enumerate(mech.bodies)   
        xinds, vinds, qinds, ωinds = fullargsinds(i)   
        setPosition!(body; x = SVector{3}(z[xinds]), q = UnitQuaternion(z[qinds]...))
        setVelocity!(body; v = SVector{3}(z[vinds]), ω = SVector{3}(z[ωinds]))
    end
end

""" Parameters """
timeStep = 10.0 # seconds

""" Define the base """
length1 = 0.5
width, depth = 0.5, 0.5
# Corner vectors
corners = [
    [[length1 / 2;length1 / 2;-length1 / 2]]
    [[length1 / 2;-length1 / 2;-length1 / 2]]
    [[-length1 / 2;length1 / 2;-length1 / 2]]
    [[-length1 / 2;-length1 / 2;-length1 / 2]]
    [[length1 / 2;length1 / 2;length1 / 2]]
    [[length1 / 2;-length1 / 2;length1 / 2]]
    [[-length1 / 2;length1 / 2;length1 / 2]]
    [[-length1 / 2;-length1 / 2;length1 / 2]]
]
# Initial orientation
##
ϕ1 = 0;
q1 = UnitQuaternion(RotZ(ϕ1))

# Define base link
origin = Origin{Float64}()
link0 = Box(width, depth, length1, 1., color = RGBA(1., 1., 0.))
link0.m = 10 # set base mass

# Constraints on base
# TODO: find what "Friction()" is doing
impacts = [InequalityConstraint(Friction(link0,[0;0;1.0], 0.01; p = corners[i])) for i=1:8] # above ground
world2base = EqualityConstraint(Floating(origin, link0)) # free floating

""" Define arms """
arm_length = 0.1
arm_width = 1.0
arm_depth = 0.1

# Define arm link
link1 = Box(arm_width, arm_depth, arm_length, arm_length, color = RGBA(0., 1., 0.))

# Constraints on the arms
joint1_axis = [0;0;1] # joint 1 rotates about z axis

vert01 = [width/2; 0; 0] # connection offset from link0 to joint1
vert11 = [-arm_width/2; 0; 0] # connection offset from link1 to joint1

joint1 = EqualityConstraint(Revolute(link0, link1, joint1_axis; p1=vert01,p2=vert11)) # joint1 : base to link1

# put them together
links = [link0; link1]
eqcs = [world2base; joint1]
ineqcs = impacts

mech = Mechanism(origin, links, eqcs, ineqcs) # TODO: this function is mutating!!!
# setPosition!(link0, x = [0.;0.;1.])
# setVelocity!(link0, v = [0;0;0], ω = [0;0;0])
# setPosition!(link1, x = [width/2;0.;1.])
# setVelocity!(link1, v = [0;0;0], ω = [0;0;0])


""" test state """
x0 = generate_config(mech, [0.0;0.0;1.0;0.0], [pi/3]);
reshape(x0,(13,2))'
# visualize state 
setStates!(mech,x0)
reshape(round.(getStates(mech)',digits=3),(13,2))'
steps = Base.OneTo(1)
storage = CD.Storage{Float64}(steps,length(mech.bodies))
for i=1:2
    storage.x[i][1] = mech.bodies[i].state.xc
    storage.v[i][1] = mech.bodies[i].state.vc
    storage.q[i][1] = mech.bodies[i].state.qc
    storage.ω[i][1] = mech.bodies[i].state.ωc
end
visualize(mech,storage, env = "editor")

""" Define flotation force through controller """
baseid = eqcs[1].id # this is the floating base, as a free joint
mech.g = 0 # disable gravity
function controller!(mechanism, k)
    # F = SA[0;0; 9.81 * 10.4]
    F = SA[0;0; 0]
    τ = SA[0;0;0.0]
    setForce!(mechanism, geteqconstraint(mechanism,baseid), [F;τ])
    return
end
""" Start simulation """
# storage = simulate!(mech, timeStep, record = true)
# visualize(mech, storage, env = "editor")


