
using RigidBodyDynamics # Need to install latest version without tag
const RBD = RigidBodyDynamics
include("floatingBaseSpace.jl")
using RobotDynamics
using StaticArrays

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

struct FloatingSpaceRBD{T} <: AbstractModel
    body_mass::T
    body_size::T
    arm_mass::T
    arm_width::T
    arm_length::T 
    body_mass_mtx::Array{T,2}
    arm_mass_mtx::Array{T,2}
    body_inertias::Diagonal{T,Array{T,1}}
    arm_inertias::Diagonal{T,Array{T,1}}
    _joint_directions::Array{Array{T,1},1}
    g::T 
    nb::Integer     # number of arm links, total rigid bodies in the system will be nb+1
    tree::RigidBodyDynamics.Mechanism{Float64}
    function FloatingSpaceRBD{T}(nb, _joint_directions) where {T<:Real} 
        g = 0      # in space, no gravity!
        body_mass = 10.0
        body_size = 0.5
        arm_mass = 1.0
        arm_width = 0.1
        arm_length = 1.0
        body_mass_mtx = diagm([body_mass,body_mass,body_mass])
        arm_mass_mtx = diagm([arm_mass,arm_mass,arm_mass])
        body_inertias = Diagonal(1 / 12 * body_mass * diagm([0.5^2 + 0.5^2;0.5^2 + 0.5^2;0.5^2 + 0.5^2]))
        arm_inertias = Diagonal(1 / 12 * arm_mass * diagm([0.1^2 + 0.1^2;0.1^2 + 1.0^2;1.0^2 + 0.1^2]))


        # define zero gravity world
        world = RBD.RigidBody{Float64}("world")
        FloatingSpaceRobot = RBD.Mechanism(world; gravity=[0., 0., 0.])
    
        joint0 = RBD.Joint("joint0", RBD.QuaternionFloating{Float64}()) # joint connecting base to world
        # inertia frame of base
        base_inertia_frame = RBD.SpatialInertia(
            RBD.frame_after(joint0), 
            com = [0.0, 0.0, 0.0], # center of mass location with respect to joint, should be all zero for the base
            moment_about_com = body_inertias, # inertia matirx
            mass = body_mass
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
        joint[1] = RBD.Joint(string("joint", 1), RBD.Revolute(_joint_directions[1]))
        inertia_frame = RBD.SpatialInertia(
            RBD.frame_after(joint[1]), 
            com = [arm_length/2, 0, 0], # center of mass location with respect to joint
            # arm initialized at positive x direction
            moment_about_com = arm_inertias, # inertia matirx
            mass = arm_mass
        )
        link[1] = RBD.RigidBody(string("link", 1), inertia_frame)
        joint_pose = RBD.Transform3D(
                RBD.frame_before(joint[1]), 
                RBD.frame_after(joint0), # base frame
                one(RotMatrix{3}), 
                SVector{3}([body_size/2, 0., 0.]) # joint1 is off by half of the body size to positive x axis
            )
        RBD.attach!(FloatingSpaceRobot, base, link[1], joint[1], joint_pose = joint_pose)
    
        for idx = 2 : nb
            joint[idx] = RBD.Joint(string("joint", idx), RBD.Revolute(_joint_directions[idx]))
            inertia_frame = RBD.SpatialInertia(
                RBD.frame_after(joint[idx]), 
                com = [arm_length/2, 0, 0], # center of mass location with respect to joint
                # arm initialized at positive x direction
                moment_about_com = arm_inertias, # inertia matirx
                mass = arm_mass
            )
            link[idx] = RBD.RigidBody(string("link", idx), inertia_frame)
            joint_pose = RBD.Transform3D(
                RBD.frame_before(joint[idx]), 
                RBD.frame_after(joint[idx - 1]), # base frame
                one(RotMatrix{3}), 
                SVector{3}([arm_length, 0., 0.]) # next joint is off by half of the arm length to positive x axis
            )
            RBD.attach!(FloatingSpaceRobot, link[idx-1], link[idx], joint[idx], joint_pose = joint_pose)
        end
        new(body_mass,
            body_size,
            arm_mass,
            arm_width,
            arm_length,
            body_mass_mtx,
            arm_mass_mtx,
            body_inertias,
            arm_inertias,
            _joint_directions,
            g,
            nb,
            FloatingSpaceRobot
        )    
    end   
end

function FloatingSpaceOrthRBD(nb::Integer)
    _joint_directions = [ i % 2 == 0 ? [0.0;1.0;0.0] : [0.0;0.0;1.0] for i=1:nb]
    FloatingSpaceRBD{Float64}(nb, _joint_directions)
end

# Specify the state and control dimensions
function RobotDynamics.state_dim(model::FloatingSpaceRBD)
    7 + model.nb * 2
end
function RobotDynamics.control_dim(model::FloatingSpaceRBD)
    6 + model.nb
end

# TODO: implement dynamics
function RobotDynamics.dynamics(model::DoublePendulumRC, x, u)
    1
end

# Create the model
model = FloatingSpaceOrthRBD(3)
n,m = size(model)

# Generate random state and control vector
x,u = rand(model)
dt = 0.01
z = KnotPoint(x,u,dt)

# Evaluate the continuous dynamics and Jacobian
ẋ = dynamics(model, x, u)
∇f = RobotDynamics.DynamicsJacobian(model)   # only allocate memory
jacobian!(∇f, model, z)   # calls jacobian in integration.jl

# Evaluate the discrete dynamics and Jacobian
x′ = discrete_dynamics(RK3, model, z)
discrete_jacobian!(RK3, ∇f, model, z)
println(x′)