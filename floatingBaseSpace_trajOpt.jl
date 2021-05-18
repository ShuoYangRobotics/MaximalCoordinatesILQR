import Pkg; Pkg.activate(joinpath(@__DIR__)); 
Pkg.instantiate();
include("floatingBaseSpace.jl")
include("floatingBaseSpace_test.jl")
using Profile
using TimerOutputs
model = FloatingSpaceOrth(3)

# run test to trigger model function compile
test_dyn()

# put solve steps in function 
function solve_altro_test(model)
    # trajectory 
    N = 100   
    dt = 0.005                  # number of knot points
    tf = (N-1)*dt           # final time
    n,m = size(model)

    U0 = @SVector fill(0.00001, m)
    U_list = [U0 for k = 1:N-1]

    x0 = generate_config(model, [0.01;0.01;0.01;0.01], fill.(0.01,model.nb))

    xf = generate_config(model, [0.3;0.3;1.0;pi/6], fill.(pi/6,model.nb))

    # objective
    Qf = Diagonal(@SVector fill(550., n))
    Q = Diagonal(@SVector fill(1e-2, n))
    R = Diagonal(@SVector fill(1e-3, m))
    costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1e-1) for i=1:N]
    costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=550.0)
    obj = Objective(costfuns);

    # constraints
    # Create Empty ConstraintList
    conSet = ConstraintList(n,m,N)
    
    #  limit the velocity of the last link 
    # index of the last link
    # vmax = 5.0
    # statea_inds!(model, model.nb+1)
    # p = 3
    # A = zeros(p,n+m)
    # A[:,model.v_ainds] = I(3)
    # b = zeros(p)
    # b .= vmax
    # b2 = zeros(p)
    # b2 .= -vmax

    # lin_lower = LinearConstraint(n,m,A,b, Inequality())
    # lin_upper = LinearConstraint(n,m,-A,-b2, Inequality())

    # add_constraint!(conSet, lin_lower, 1:N-1)
    # add_constraint!(conSet, lin_upper, 1:N-1)

    # u_max = vcat(0.0003*(@SVector ones(2)),Inf*(@SVector ones(m-2)))
    # u_min = vcat(-0.0003*(@SVector ones(2)),-Inf*(@SVector ones(m-2)))
    # x_max=Inf*(@SVector ones(n))
    # x_min=-Inf*(@SVector ones(n))
    # control_bound = BoundConstraint(n, m; x_min, x_max, u_min, u_max)
    # add_constraint!(conSet, control_bound, 1:N-1)
    
    to = TimerOutput()
    # # problem
    prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet);

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

altro = solve_altro_test(model)

n,m = size(model)
N = 100
dt = 0.005 
x0 = generate_config(model, [0.01;0.01;0.01;0.01], fill.(0.01,model.nb))
X_list = states(altro)

mech = vis_mech_generation(model)
steps = Base.OneTo(Int(N))
storage = CD.Storage{Float64}(steps,length(mech.bodies))
for idx = 1:N
    setStates!(model,mech,X_list[idx])
    for i=1:model.nb+1
        storage.x[i][idx] = mech.bodies[i].state.xc
        storage.v[i][idx] = mech.bodies[i].state.vc
        storage.q[i][idx] = mech.bodies[i].state.qc
        storage.ω[i][idx] = mech.bodies[i].state.ωc
    end
end
visualize(mech,storage, env = "editor")

U_list = controls(altro)

using Plots
# plot all controls
plot(1:N-1, U_list)

# a final simulation pass to get "real" state trajectory
λ_init = zeros(5*model.nb)
λ = λ_init
Xfinal_list = copy(X_list)

Xfinal_list[1] = SVector{n}(x0)
for idx = 1:N-1
    println("step: ",idx)
    x1, λ1 = discrete_dynamics(model,Xfinal_list[idx], U_list[idx], λ, dt)
    println(norm(fdyn(model,x1, Xfinal_list[idx], U_list[idx], λ1, dt)))
    println(norm(g(model,x1)))
    setStates!(model,mech,x1)
    Xfinal_list[idx+1] = SVector{n}(x1)
    λ = λ1
end

# plot velocity of the last link
statea_inds!(model, model.nb+1)
# p = [Xfinal_list[i][model.v_ainds[1]] for i=1:N]
# plot(1:N, p)
# p = [Xfinal_list[i][model.v_ainds[2]] for i=1:N]
# plot!(1:N, p)
p = zeros(N,3)
for dim=1:3
p[:,dim] .= [Xfinal_list[i][model.v_ainds[dim]] for i=1:N]
end
plot(1:N, p)

