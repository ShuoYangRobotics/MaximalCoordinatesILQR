import Pkg;
Pkg.activate(joinpath(@__DIR__,"..","..")); Pkg.instantiate();
using TimerOutputs
using Test
include("../../src/RBD_floatingBase.jl")
# TODO: find out which function is required in MC version to run Altro
include("../../src/MC_floatingBase.jl")
"""Constants"""
##
Tf = 0.5
dt = 0.005
N = Int(Tf/dt)
ArmNumber = 3
vMax = 5.0
"""Generate model"""
RBDmodel = FloatingSpaceOrthRBD(ArmNumber)

"""Run Altro"""
function solve_altro_test(RBDmodel, dt, N, vmax)
    n,m = size(RBDmodel)
    n̄ = state_diff_size(RBDmodel)
    # trajectory 
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

    to = TimerOutput()


    # constraints
    # Create Empty ConstraintList
    conSet = ConstraintList(n,m,N)
    vel_limit = EFVConstraint(n,m,RBDmodel,vmax, TO.Inequality())
    add_constraint!(conSet, vel_limit, 1:N-1)
    
    # problem
    prob = Problem(RBDmodel, obj, xf, tf, x0=x0, constraints=conSet);

    initial_controls!(prob, U_list)
    opts = SolverOptions(verbose=7, 
        static_bp=0, 
        square_root = true,
        iterations=1000, bp_reg=true,
        dJ_counter_limit = 1,
        iterations_inner = 30,
        cost_tolerance=1e-4, constraint_tolerance=1e-4,
        show_summary=true)
    altro = ALTROSolver(prob, opts)
    set_options!(altro, show_summary=true)
    solve!(altro);
    return altro
end
altro = solve_altro_test(RBDmodel, dt, N, vMax)
# run it twice to get execution time
altro = solve_altro_test(RBDmodel, dt, N, vMax)

"""Visualization"""
n,m = size(RBDmodel)
X_list = states(altro)
U_list = controls(altro)
qs = []
vs = []
for idx = 1:N
    push!(qs,X_list[idx][1:(7 + ArmNumber)])
    push!(vs,X_list[idx][(8 + ArmNumber):end])
end
view_sequence(RBDmodel, qs, vs)

"""Plot and save results"""

using Plots
result_path = "results/3.vel_constraint/"
file_name = "RBD_vel_constraint_"*string(ArmNumber)*"Arms_"*string(Int(floor(vMax)))

# plot velocity of the last link 
velocity_list = zeros(N,3)
for idx=1:N
    velocity_list[idx,:] .= world_vel(RBDmodel, [X_list[idx];zeros(m)]) 
end
plot(1:N, velocity_list,title = "End Effector velocity", labels = ["x" "y" "z"],fmt = :png)
xlabel!("Time step")
ylabel!("World frame velocity")
savefig(result_path*file_name)

# save altro stats
using JLD
save(result_path*file_name*".jld", 
    "X_list", X_list, 
    "U_list", U_list,
    "Total_iter", altro.stats.iterations,
    "Solve_time", altro.stats.tsolve,
    "Cost_hist", altro.stats.cost,
    "Solve_status", altro.stats.status)
Solve_status = load(result_path*file_name*".jld", "Solve_status")
