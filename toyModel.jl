import Pkg; Pkg.activate(joinpath(@__DIR__)); 
Pkg.instantiate();


using ConstrainedDynamics
using ConstrainedDynamicsVis
using StaticArrays

timeStep = 10.0 # seconds
# Parameters
# TODO: figure out how to set mass

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
q1 = UnitQuaternion(RotX(ϕ1))

# Define base link
origin = Origin{Float64}()
link0 = Box(width, depth, length1, 1., color = RGBA(1., 1., 0.))

# Constraints on base
# TODO: find what "Friction()" is doing
impacts = [InequalityConstraint(Friction(link0,[0;0;1.0], 0.2; p = corners[i])) for i=1:8] # above ground
world2base = EqualityConstraint(Floating(origin, link0)) # free floating

""" Define arms """
arm_length = 0.1
arm_width = 1.0
arm_depth = 0.1

# Define arm link
link1 = Box(arm_width, arm_depth, arm_length, arm_length, color = RGBA(0., 1., 0.))
link2 = Box(arm_width, arm_depth, arm_length, arm_length, color = RGBA(0., 1., 1.))
link3 = Box(arm_width, arm_depth, arm_length, arm_length, color = RGBA(0., 1., 0.))
link4 = Box(arm_width, arm_depth, arm_length, arm_length, color = RGBA(0., 1., 1.))

# Constraints on the arms
joint1_axis = [0;0;1] # joint 1 rotates about z axis
joint2_axis = [0;0;1] # joint 2 rotates about z axis
joint3_axis = [0;0;1] # joint 1 rotates about z axis
joint4_axis = [0;0;1] # joint 2 rotates about z axis

vert01 = [width/2; 0; 0] # connection offset from link0 to joint1
vert11 = [-arm_width/2; 0; 0] # connection offset from link1 to joint1
vert12 = [arm_width/2; 0; 0] # connection offset from link1 to joint2
vert22 = [-arm_width/2; 0; 0] # connection offset from link2 to joint2

vert03 = [-width/2; 0; 0] # connection offset from link0 to joint3
vert33 = [arm_width/2; 0; 0] # connection offset from link3 to joint3
vert34 = [-arm_width/2; 0; 0] # connection offset from link3 to joint4
vert44 = [arm_width/2; 0; 0] # connection offset from link4 to joint4

joint1 = EqualityConstraint(Revolute(link0, link1, joint1_axis; p1=vert01,p2=vert11)) # joint1 : base to link1
joint2 = EqualityConstraint(Revolute(link1, link2, joint2_axis; p1=vert12,p2=vert22)) # joint2: link1 to link2

joint3 = EqualityConstraint(Revolute(link0, link3, joint3_axis; p1=vert03,p2=vert33)) # joint1 : base to link3
joint4 = EqualityConstraint(Revolute(link3, link4, joint4_axis; p1=vert34,p2=vert44)) # joint2: link3 to link4

# put them together
links = [link0; link1; link2; link3; link4]
eqcs = [world2base; joint1; joint2; joint3; joint4]
ineqcs = impacts

mech = Mechanism(origin, links, eqcs, ineqcs) # TODO: this function is mutating!!!
setPosition!(link0, x = [0.;0.;1.])
setVelocity!(link0, v = [0;0;0], ω = [0;0;0])
setPosition!(link1, x = [width/2;0.;1.])
setVelocity!(link1, v = [0;0;0], ω = [0;0;0])
setPosition!(link2, x = [width + arm_width/2;0.;1.])
setVelocity!(link2, v = [0;0;0], ω = [0;0;0])
setPosition!(link3, x = [-width/2;0.;1.])
setVelocity!(link3, v = [0;0;0], ω = [0;0;0])
setPosition!(link4, x = [-width - arm_width/2;0.;1.])
setVelocity!(link4, v = [0;0;0], ω = [0;0;0])

""" Define flotation force through controller """
baseid = eqcs[1].id # this is the floating base, as a free joint
mech.g = 0 # disable gravity
function controller!(mechanism, k)
    F = SA[0;0;0]
    τ = SA[0;0;0.01]
    setForce!(mechanism, geteqconstraint(mechanism,baseid), [F;τ])
    return
end

""" Start simulation """
storage = simulate!(mech, timeStep, controller!, record = true)
visualize(mech, storage, env = "editor")


