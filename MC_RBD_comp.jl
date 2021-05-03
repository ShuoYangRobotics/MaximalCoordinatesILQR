import Pkg; Pkg.activate(joinpath(@__DIR__)); 
Pkg.instantiate();
include("floatingBaseSpace.jl")
include("RBDmodel.jl")

"""Time Constants"""
Tf = 5
dt = 0.005
N = Int(Tf/dt)

"""Build MC model"""
ArmNumber = 6
model = FloatingSpaceOrth(ArmNumber)

"""Build RBD model"""
#generate RBD model from MC model
function generate_RBD_model(model::FloatingSpace)
    nb = model.nb # the number of arms
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
    
    """Adding arms"""
    joint = Array{Joint{Float64,RigidBodyDynamics.Revolute{Float64}}}(UndefInitializer(), nb)
    link = Array{RigidBodyDynamics.RigidBody{Float64}}(UndefInitializer(), nb)

    # first arm linking to base
    joint[1] = RBD.Joint(string("joint", 1), RBD.Revolute(model._joint_directions[1]))
    inertia_frame = RBD.SpatialInertia(
        RBD.frame_after(joint[1]), 
        com = [model.arm_length/2, 0, 0], # center of mass location with respect to joint
        # arm initialized at positive x direction
        moment_about_com = model.arm_inertias, # inertia matirx
        mass = model.arm_mass
    )
    link[1] = RBD.RigidBody(string("link", 1), inertia_frame)
    joint_pose = RBD.Transform3D(
            RBD.frame_before(joint[1]), 
            RBD.frame_after(joint0), # base frame
            one(RotMatrix{3}), 
            SVector{3}([model.body_size/2, 0., 0.]) # joint1 is off by half of the body size to positive x axis
        )
    RBD.attach!(FloatingSpaceRobot, base, link[1], joint[1], joint_pose = joint_pose)

    for idx = 2 : nb
        joint[idx] = RBD.Joint(string("joint", idx), RBD.Revolute(model._joint_directions[idx]))
        inertia_frame = RBD.SpatialInertia(
            RBD.frame_after(joint[idx]), 
            com = [model.arm_length/2, 0, 0], # center of mass location with respect to joint
            # arm initialized at positive x direction
            moment_about_com = model.arm_inertias, # inertia matirx
            mass = model.arm_mass
        )
        link[idx] = RBD.RigidBody(string("link", idx), inertia_frame)
        joint_pose = RBD.Transform3D(
            RBD.frame_before(joint[idx]), 
            RBD.frame_after(joint[idx - 1]), # base frame
            one(RotMatrix{3}), 
            SVector{3}([model.arm_length, 0., 0.]) # next joint is off by half of the arm length to positive x axis
        )
        RBD.attach!(FloatingSpaceRobot, link[idx-1], link[idx], joint[idx], joint_pose = joint_pose)
    end
    return FloatingSpaceRobot
end
# FloatingSpaceRobot = generate_RBD_model(model::FloatingSpace)

# define from scratch
RBDmodel = FloatingSpaceOrthRBD(ArmNumber)
FloatingSpaceRobot = RBDmodel.tree

"""Build visualizer"""
# using MC implementation
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

    """Arm link"""
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
# we are still not handling velocities well
generate_config_rc2mc(model, base_x0, base_q0, base_v0, base_ω0, joint_angles0)
@test x0 ≈ generate_config_rc2mc(model, base_x0, base_q0, base_v0, base_ω0, joint_angles0)

# Visualize initial condition
view_single_state(model, x0)

"""Input sequence"""
U = 0.03*zeros(6+model.nb)
U[5] = 1
U[6] = 1
@show U
"""MC simulation"""
x = x0
λ_init = zeros(5*model.nb)
λ = λ_init
for idx = 1:N
    # println("step: ",idx)
    x1, λ1 = discrete_dynamics(model,x, U, λ, dt)
    # println(norm(fdyn(model,x1, x, U, λ1, dt)))
    # println(norm(g(model,x1)))
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
state = RBD.MechanismState(FloatingSpaceRobot)
RBD.set_configuration!(state, joints(FloatingSpaceRobot)[1], base_q0) # set floating joint rotation
RBD.set_velocity!(state, joints(FloatingSpaceRobot)[1], [base_ω0; base_v0])

for idx = 1 : model.nb  
    RBD.set_configuration!(state, joints(FloatingSpaceRobot)[idx+1], joint_angles0[idx])
    RBD.set_velocity!(state, joints(FloatingSpaceRobot)[idx+1], 0)
end

RBD.set_configuration!(state, RBD.configuration(state) .+ [zeros(4);base_x0;zeros(model.nb)])

controller! = function controller!(τ, t, state)
    τ .= 0.03*zeros(6 + model.nb)
    τ[2] = U[5]
    τ[3] = U[6]
end

ts, qs, vs = RBD.simulate(state, Tf,controller!, Δt=dt)

for idx = 1:N
    current_state = generate_config_rc2mc(model, qs[idx][5:7], qs[idx][1:4], vs[idx][4:6], vs[idx][1:3], qs[idx][8:end])
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

@test storage.x[4][100] ≈ storage_RBD.x[4][100] atol = 1e-3

