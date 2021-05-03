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
# generate from MC model
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

