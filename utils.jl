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