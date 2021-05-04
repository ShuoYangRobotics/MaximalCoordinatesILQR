import Pkg; Pkg.activate(joinpath(@__DIR__)); 
Pkg.instantiate();
using BenchmarkTools
# we study the speed and memory allocation of different method
x = [1.0;2;3;4;5;6]
vec1 = zeros(3)
tmp = [SA[1.0,i,1.0] for i=1:10]

function test_static(x,vec1,tmp_data)
    tmp_data[1] = SVector{3}(x[1:3])
    tmp_data[2] = SVector{3}(x[4:6])
    vec1 .= cross(tmp_data[1],tmp_data[2]) + 3*tmp_data[1]
    return 
end



tmp2 = [SizedVector{3}(zeros(3)) for i=1:10]
tmp_mat33 = SizedMatrix{3,3}(zeros(3,3))
function hat!(mtx, vec)
    mtx .= 0
    mtx[1,2] = - vec[3]
    mtx[1,3] =  vec[2]
    mtx[2,1] =   vec[3]
    mtx[2,3] =  - vec[1]
    mtx[3,1] =  - vec[2]
    mtx[3,2] =  vec[1]
    return
end
function test_sized(x,vec1,tmp_data, tmp_mat33)
    tmp_data[1][1] = x[1]
    tmp_data[1][2] = x[2]
    tmp_data[1][3] = x[3]
    tmp_data[2][1] = x[4]
    tmp_data[2][2] = x[5]
    tmp_data[2][3] = x[6]
    # vec1 .= cross(tmp_data[1],tmp_data[2])
    hat!(tmp_mat33, tmp_data[1]) 
    mul!(vec1, tmp_mat33, tmp_data[2])
    vec1[1] += tmp_data[1][1]*3
    vec1[2] += tmp_data[1][2]*3
    vec1[3] += tmp_data[1][3]*3
    return 
end


tmp3 = [zeros(3) for i=1:10]
tmp2_mat33 = zeros(3,3)

vec2 = copy(vec1)
vec3 = copy(vec1)
@btime test_static(x,vec1, tmp)                   # 64.335 ns (2 allocations: 224 bytes)
@btime test_sized(x,vec2, tmp2, tmp_mat33)        # 36.575 ns (0 allocations: 0 bytes)
@btime test_sized(x,vec3, tmp3, tmp2_mat33)       # 57.818 ns (0 allocations: 0 bytes)

# the best solution seems to use sized array, preallocation space, avoid memory allocation
