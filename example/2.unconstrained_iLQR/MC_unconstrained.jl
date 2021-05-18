import Pkg;
Pkg.activate(joinpath(@__DIR__,"..","..")); Pkg.instantiate();
using TimerOutputs
using Test
include("../../src/MC_floatingBase.jl")

"""Constants"""
Tf = 0.5
dt = 0.005
N = Int(Tf/dt)
ArmNumber = 1

"""Generate model"""
MCmodel = FloatingSpaceOrth(ArmNumber)

# run test to trigger model function compile
test_dyn()

# put solve steps in function 
function solve_altro_test(model, N, dt)
    # trajectory 
    tf = (N-1)*dt           # final time
    n,m = size(model)

    U0 = @SVector fill(0.00001, m)
    U_list = [U0 for k = 1:N-1]

    x0 = generate_config(model, [0.01;0.01;0.01;0.01], fill.(0.01,model.nb))

    xf = generate_config(model, [0.3;0.3;1.0;pi/6], fill.(pi/6,model.nb))

    # objective
    Qf = Diagonal(@SVector fill(550., n))
    Q = Diagonal(@SVector fill(1e-1, n))
    R = Diagonal(@SVector fill(1e-4, m))
    costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1) for i=1:N]
    costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=550.0)
    obj = Objective(costfuns);

    # constraints
    # Create Empty ConstraintList
    conSet = ConstraintList(n,m,N)
    
    to = TimerOutput()
    # # problem
    prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet);

    initial_controls!(prob, U_list)
    opts = SolverOptions(verbose=7, 
        static_bp=0, 
        square_root = true,
        iterations=150, bp_reg=true,
        constraint_force_reg = 0.0,
        line_search_coefficient = 2.0,
        dJ_counter_limit = 1,
        iterations_inner = 30,
        cost_tolerance=1e-4, constraint_tolerance=1e-4)
    altro = ALTROSolver(prob, opts)
    set_options!(altro, show_summary=true)
    solve!(altro);
    return altro
end
altro = solve_altro_test(MCmodel, N, dt)
# run it twice to get execution time
altro = solve_altro_test(MCmodel, N, dt)

"""Visualization"""
n,m = size(MCmodel)
X_list = states(altro)
U_list = controls(altro)

# a final simulation pass to get "real" state trajectory
λ_init = zeros(5*MCmodel.nb)
λ = λ_init
Xfinal_list = copy(X_list)
Xfinal_list[1] = SVector{n}(X_list[1])
mech = vis_mech_generation(MCmodel)
for idx = 1:N-1
    x1, λ1 = discrete_dynamics(MCmodel,Xfinal_list[idx], U_list[idx], λ, dt)
    setStates!(MCmodel,mech,x1)
    Xfinal_list[idx+1] = SVector{n}(x1)
    λ = λ1
end
view_sequence(MCmodel, Xfinal_list)

"""Plot and save results"""

using Plots
result_path = "results/2.unconstrained_iLQR/"
file_name = "MC_unconstrained_"*string(ArmNumber)*"Arms"

# plot velocity of the last link
statea_inds!(MCmodel, MCmodel.nb+1)
p = zeros(N,3)
for dim=1:3
    p[:,dim] .= [Xfinal_list[i][MCmodel.v_ainds[dim]] for i=1:N]
end
plot(1:N, p, title = "End Effector velocity", labels = ["x" "y" "z"],fmt = :png)
xlabel!("Time step")
ylabel!("World frame velocity")
savefig(result_path*file_name)

Max_vel = max(abs.(p)...)

# save altro stats
using JLD
save(result_path*file_name*".jld", 
    "X_list", X_list, 
    "U_list", U_list,
    "Total_iter", altro.stats.iterations,
    "Solve_time", altro.stats.tsolve,
    "Cost_hist", altro.stats.cost,
    "Solve_status", altro.stats.status,
    "Max_vel", Max_vel)
Solve_status = load(result_path*file_name*".jld", "Solve_status")
