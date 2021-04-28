import Pkg; Pkg.activate(joinpath(@__DIR__)); 
Pkg.instantiate();
include("floatingBaseSpace.jl")
using RigidBodyDynamics # Need to install latest version without tag
const RBD = RigidBodyDynamics
"""Time Constants"""
Tf = 0.5
dt = 0.005
N = Int(Tf/dt)

"""Build visualizer"""
# using MC implementation
model = FloatingSpaceOrth(2)
mech = vis_mech_generation(model)
steps = Base.OneTo(Int(N))
storage = CD.Storage{Float64}(steps,length(mech.bodies))
storage_RBD = CD.Storage{Float64}(steps,length(mech.bodies))
function view_single_state(model::FloatingSpace, x)
    mech = vis_mech_generation(model)
    storage_single = CD.Storage{Float64}(Base.OneTo(1),length(mech.bodies))
    setStates!(model, mech, x)
    CD.discretizestate!(mech)
    for i=1:model.nb+1
        storage_single.x[i][1] = mech.bodies[i].state.xc
        storage_single.v[i][1] = mech.bodies[i].state.vc
        storage_single.q[i][1] = mech.bodies[i].state.qc
        storage_single.ω[i][1] = mech.bodies[i].state.ωc
    end
    visualize(mech, storage_single, env = "editor")
end

"""Initial conditions"""
function generate_config_rc2mc(
    model::FloatingSpace, 
    base_translation, 
    base_rotations,
    base_v,
    base_ω,
    joint_angles)
    pin = zeros(3) # com position of the body link 
    """Base"""
    pin[1] = base_translation[1]
    pin[2] = base_translation[2]
    pin[3] = base_translation[3]
    prev_q = UnitQuaternion(base_rotations...)
    state = [pin;base_v;RS.params(prev_q);base_ω]

    """Arm links"""
    pin = pin+prev_q * [model.body_size/2,0,0]
    # find quaternion from joint angles
    rotations = []
    @assert length(joint_angles) == model.nb
    for i=1:length(joint_angles)
        axis = model.joint_directions[i]
        push!(rotations, 
            UnitQuaternion(AngleAxis(joint_angles[i], axis[1], axis[2], axis[3]))
        )
    end
    for i = 1:length(rotations)
        r = UnitQuaternion(rotations[i])
        link_q = prev_q * r
        delta = link_q * [model.arm_length/2,0,0] # assume all arms have equal length
        link_x = pin+delta
        state = [state; link_x;zeros(3);RS.params(link_q);zeros(3)]
        # arm velocities can be calculated but doesn't matter for visualization
        # TODO: calculate arm velocities
        prev_q = link_q
        pin += 2*delta
    end
    return state
end
base_x0 = [0., 0., 1.]
base_q0 = RS.params(UnitQuaternion(RotZ(pi/3)))
base_v0 = [0., 0., 0.]
base_ω0 = [0., 0., 0.]
joint_angles0 = fill.(pi/13,model.nb)

x0 = generate_config(model, [base_x0;pi/3], joint_angles0);
@test x0 ≈ generate_config_rc2mc(model, base_x0, base_q0, base_v0, base_ω0, joint_angles0)

# Visualize initial condition
view_single_state(model, x0)

"""Input sequence"""
U = 0.03*zeros(6+model.nb)
U[5] = 1
U[6] = 1
"""MC simulation"""
x = x0
λ_init = zeros(5*model.nb)
λ = λ_init
for idx = 1:N
    println("step: ",idx)
    x1, λ1 = discrete_dynamics(model,x, U, λ, dt)
    println(norm(fdyn(model,x1, x, U, λ1, dt)))
    println(norm(g(model,x1)))
    setStates!(model,mech,x1)
    for i=1:model.nb+1
        storage.x[i][idx] = mech.bodies[i].state.xc
        storage.v[i][idx] = mech.bodies[i].state.vc
        storage.q[i][idx] = mech.bodies[i].state.qc
        storage.ω[i][idx] = mech.bodies[i].state.ωc
    end
    x = x1
    λ = λ1
end
visualize(mech,storage, env = "editor")

"""RBD simulation"""
# define zero gravity world
world = RBD.RigidBody{Float64}("world")
FloatingSpaceRobot = RBD.Mechanism(world; gravity=[0., 0., 0.])

joint0 = RBD.Joint("joint0", RBD.QuaternionFloating{Float64}()) # joint connecting base to world

# inertia frame of base
base_inertia_frame = RBD.SpatialInertia(
    RBD.frame_after(joint0), 
    com = [0.0, 0.0, 0.0], # center of mass location with respect to joint, should be all zero for the base
    moment_about_com = model.body_inertias, # inertia matirx
    mass = model.body_mass
)
base = RBD.RigidBody("base", base_inertia_frame) # define body "base" using frame "base_inertia_frame"

# this describes the position of joint 0 relative to the world
before_joint0_to_world = RBD.Transform3D(
    RBD.frame_before(joint0), # frame before joint 0 should be world frame
    RBD.default_frame(world), # world is a body, this gets the world's body frame
    one(RotMatrix{3}), # now we just set rotation and translation as unit/zero
    SVector{3}([0., 0., 0.])
)
# now we connect "base" to "world" through "joint0" which is located at "before_joint0_to_world"
RBD.attach!(FloatingSpaceRobot, world, base, joint0, joint_pose=before_joint0_to_world)

@testset "simulate base" begin
    state = RBD.MechanismState(FloatingSpaceRobot)
    q0 = base_q0
    v0 = [0., 0., 1.]
    ω0 = [1., 0., 0.]
    RBD.set_configuration!(state, joint0, q0) # set floating joint rotation
    RBD.set_velocity!(state, joint0, [ω0; v0])
    
    ts, qs, vs = simulate(state, 1., Δt=0.001)
    @test all(all(!isnan, q) for q in qs)
    @test all(all(!isnan, v) for v in vs)
    @test all(norm(v[4:6])≈ 1.0 for v in vs) 
end

# Add the first arm link
joint1 = RBD.Joint("joint1", RBD.Revolute(model.joint_directions[1])) # joint1 rotate about axis "model.joint_directions[1]"
link1_inertia_frame = RBD.SpatialInertia(
    RBD.frame_after(joint1), 
    com = [model.arm_length/2, 0, 0], # center of mass location with respect to joint
    # arm initialized at positive x direction
    moment_about_com = model.arm_inertias, # inertia matirx
    mass = model.arm_mass
)
link1 = RBD.RigidBody("link1", link1_inertia_frame)

before_joint1_to_base = RBD.Transform3D(
    RBD.frame_before(joint1), 
    RBD.frame_after(joint0), # base frame
    one(RotMatrix{3}), 
    SVector{3}([model.body_size/2, 0., 0.]) # joint1 is off by half of the body size to positive x axis
)

RBD.attach!(FloatingSpaceRobot, base, link1, joint1, joint_pose = before_joint1_to_base)

# Add the second arm link
joint2 = RBD.Joint("joint2", RBD.Revolute(model.joint_directions[2])) 
link2_inertia_frame = RBD.SpatialInertia(
    RBD.frame_after(joint2), 
    com = [model.arm_length/2, 0, 0], # center of mass location with respect to joint
    # arm initialized at positive x direction
    moment_about_com = model.arm_inertias, # inertia matirx
    mass = model.arm_mass
)
link2 = RBD.RigidBody("link2", link2_inertia_frame)
before_joint2_to_after_joint1 = RBD.Transform3D(
    RBD.frame_before(joint2), 
    RBD.frame_after(joint1), # link1 frame
    one(RotMatrix{3}), 
    SVector{3}([model.body_size/2, 0., 0.])
)

RBD.attach!(FloatingSpaceRobot, link1, link2, joint2, joint_pose = before_joint2_to_after_joint1)

# start simulation
state = RBD.MechanismState(FloatingSpaceRobot)

RBD.set_configuration!(state, joint0, base_q0) # set floating joint rotation
RBD.set_velocity!(state, joint0, [base_ω0; base_v0])

RBD.set_configuration!(state, joint1, joint_angles0[1])
RBD.set_velocity!(state, joint1, 0)

RBD.set_configuration!(state, joint2, joint_angles0[2])
RBD.set_velocity!(state, joint2, 0)

RBD.set_configuration!(state, RBD.configuration(state) .+ [zeros(4);base_x0;0.;0.])

controller! = function controller!(τ, t, state)
    τ .= 0.03*zeros(6+model.nb)
    τ[2] = 1.
    τ[3] = 1.
end

ts, qs, vs = RBD.simulate(state, Tf,controller!, Δt=dt)

for idx = 1:N
    current_state = generate_config_rc2mc(model, qs[idx][5:7], qs[idx][1:4], vs[idx][4:6], vs[idx][1:3], qs[idx][8:9])
    setStates!(model, mech, current_state)
    CD.discretizestate!(mech)
    for i=1:model.nb+1
        storage_RBD.x[i][idx] = mech.bodies[i].state.xc
        storage_RBD.v[i][idx] = mech.bodies[i].state.vc
        storage_RBD.q[i][idx] = mech.bodies[i].state.qc
        storage_RBD.ω[i][idx] = mech.bodies[i].state.ωc
    end
end
visualize(mech, storage_RBD, env = "editor")

