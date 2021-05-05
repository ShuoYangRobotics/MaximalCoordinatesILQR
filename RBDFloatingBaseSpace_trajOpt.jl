import Pkg; Pkg.activate(joinpath(@__DIR__)); 
Pkg.instantiate();
using TimerOutputs
include("RBDmodel.jl")
include("floatingBaseSpace.jl")
include("RBDmodel_constraint.jl")

RBDmodel = FloatingSpaceOrthRBD(3)
""" Test Altro with constraint """

# put solve steps in function 
function solve_altro_test(RBDmodel)
    n,m = size(RBDmodel)
    n̄ = state_diff_size(RBDmodel)
    # trajectory 
    N = 100   
    dt = 0.005                  # number of knot points
    tf = (N-1)*dt           # final time

    U0 = @SVector fill(0.00001, m)
    U_list = [U0 for k = 1:N-1]

    base_x0 = [0.01, 0.01, 0.01]
    base_q0 = RS.params(UnitQuaternion(RotZ(0.01)))
    base_v0 = [0., 0., 0.]
    base_ω0 = [0., 0., 0.]
    joint_angles0 = fill.(0.01,RBDmodel.nb)
    joint_w0 = fill.(0.0,RBDmodel.nb)
    x0 = [base_q0;base_x0;joint_angles0;base_ω0;base_v0;joint_w0]

    base_xf = [0.3;0.3;1.0]
    base_qf = RS.params(UnitQuaternion(RotZ(pi/6)))
    base_vf = [0., 0., 0.]
    base_ωf = [0., 0., 0.]
    joint_anglesf = fill.(pi/6,RBDmodel.nb)
    joint_wf = fill.(0.0,RBDmodel.nb)
    xf = [base_qf;base_xf;joint_anglesf;base_ωf;base_vf;joint_wf]

    #x0 and xf are the same as floatinbBaseSpace_trajOpt

    # objective
    Qf = Diagonal(@SVector fill(550., n))
    Q = Diagonal(@SVector fill(1e-2, n))
    R = Diagonal(@SVector fill(1e-3, m))
    costfuns = [TO.LieLQRCost(RobotDynamics.LieState(RBDmodel), Q, R, SVector{n}(xf); w=1e-1) for i=1:N]
    costfuns[end] = TO.LieLQRCost(RobotDynamics.LieState(RBDmodel), Qf, R, SVector{n}(xf); w=550.0)
    obj = Objective(costfuns);

    # constraints
    # Create Empty ConstraintList
    conSet = ConstraintList(n,m,N)
    vel_limit = EFVConstraint(n,m,RBDmodel,5.0, TO.Inequality())
    add_constraint!(conSet, vel_limit, 1:N)

    to = TimerOutput()
    # # problem
    prob = Problem(RBDmodel, obj, xf, tf, x0=x0, constraints=conSet);

    initial_controls!(prob, U_list)
    opts = SolverOptions(verbose=7, 
        static_bp=0, 
        square_root = true,
        iterations=150, bp_reg=true,
        cost_tolerance=1e-4, constraint_tolerance=1e-4)
    altro = ALTROSolver(prob, opts)
    set_options!(altro, show_summary=true)
    solve!(altro);
    return altro

end



altro = solve_altro_test(RBDmodel)
n,m = size(RBDmodel)
N = 100
X_list = states(altro)
U_list = controls(altro)
mc_model = FloatingSpaceOrth(3)
mc_n,mc_m = size(mc_model)
mech = vis_mech_generation(mc_model)
steps = Base.OneTo(Int(N))
storage = CD.Storage{Float64}(steps,length(mech.bodies))
for idx=1:N
    mc_state = generate_config_rc2mc(mc_model, X_list[idx][5:7], X_list[idx][1:4], zeros(3), zeros(3), X_list[idx][8:7+mc_model.nb])
    setStates!(mc_model,mech,mc_state)
    for i=1:mc_model.nb+1
        storage.x[i][idx] = mech.bodies[i].state.xc
        storage.v[i][idx] = mech.bodies[i].state.vc
        storage.q[i][idx] = mech.bodies[i].state.qc
        storage.ω[i][idx] = mech.bodies[i].state.ωc
    end
end
visualize(mech,storage, env = "editor")


using Plots
#plot velocity of the last link 
velocity_list = zeros(N,3)
for idx=1:N
    velocity_list[idx,:] .= world_vel(RBDmodel, [X_list[idx];zeros(m)]) 
end
plot(1:N, velocity_list)