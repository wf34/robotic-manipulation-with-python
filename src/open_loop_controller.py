
import numpy as np

from pydrake.all import (
    PiecewisePose, PiecewisePolynomial, TrajectorySource, Demultiplexer,
    BsplineTrajectory,
    KinematicTrajectoryOptimization,
    PositionConstraint, OrientationConstraint,
    Solve,
    
    RotationMatrix,
    RigidTransform,
)

IIWA_DEFAULT_POSITION = [-1.57, 0.1, 0, -1.2, 0, 1.6, 0]

def handle_opt_result(result, trajopt, prog):
    if not result.is_success():
        print(dir(result))
        print(result.get_solver_id().name(), result.GetInfeasibleConstraintNames(prog), result.GetInfeasibleConstraints(prog))
        assert False, "Trajectory optimization failed"
    else:
        print("Trajectory optimization succeeded")
        return trajopt.ReconstructTrajectory(result)

def get_present_plant_position_with_inf(plant, plant_context, information=1., model_name='iiwa7'):
    iiwa_model_instance = plant.GetModelInstanceByName(model_name)
    indices = list(map(int, plant.GetJointIndices(model_instance=iiwa_model_instance)))
    n = plant.num_positions()
    plant_0 = plant.GetPositions(plant_context)
    plant_inf = np.eye(n) * 1.e-9
    for i in zip(indices):
        plant_inf[i, i] = information
    return plant_0, plant_inf

def constrain_position(plant, trajopt,
                       X_WG, target_time,
                       plant_context,
                       with_orientation=False,
                       pos_limit=0.0,
                       theta_bound_degrees=5):
    lower_translation, upper_translation = \
        X_WG.translation() - [pos_limit] * 3, X_WG.translation() + [pos_limit] * 3

    gripper_frame = plant.GetBodyByName("body").body_frame()
    pos_constraint = PositionConstraint(plant, plant.world_frame(),
                                        lower_translation,
                                        upper_translation,
                                        gripper_frame,
                                        [0, 0., 0.],
                                        plant_context)
    trajopt.AddPathPositionConstraint(pos_constraint, target_time)

    if with_orientation:
        orientation_constraint = OrientationConstraint(plant,
                                                       gripper_frame,
                                                       X_WG.rotation().inverse(),
                                                       plant.world_frame(),
                                                       RotationMatrix(),
                                                       np.radians(theta_bound_degrees),
                                                       plant_context)
        trajopt.AddPathPositionConstraint(orientation_constraint, target_time)


def solve_for_iiwa_internal_trajectory(plant, X_G, times, plant_temp_context):
    num_q = plant.num_positions()
    num_c = 5
    print('num_positions: {}; num control points: {}'.format(num_q, num_c))

    X_WGStart = X_G['initial']
    X_WGgoal = RigidTransform(X_G['initial'].rotation(), X_G['initial'].translation() + [0, 0, 0.3])
    print('T:', X_WGStart.translation(), X_WGgoal.translation())

    trajopt = KinematicTrajectoryOptimization(num_q, num_c)
    prog = trajopt.get_mutable_prog()

    q0, inf0 = get_present_plant_position_with_inf(plant, plant_temp_context)
    prog.AddQuadraticErrorCost(inf0, q0, trajopt.control_points()[:, 0])
    prog.AddQuadraticErrorCost(inf0, q0, trajopt.control_points()[:, -1])

    #q_guess =np.zeros((num_q, 1))
    #q_guess[:len(IIWA_DEFAULT_POSITION), 0] = IIWA_DEFAULT_POSITION

    #q_guess = np.linspace(q0[:, np.newaxis],
    #                      q0[:, np.newaxis],
    #                      trajopt.num_control_points()
    #    )[:, :, 0].T
    #trajopt.SetInitialGuess(BsplineTrajectory(trajopt.basis(), q_guess))

    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)

    plant_p_lower_limits = np.nan_to_num(plant.GetPositionLowerLimits(), neginf=0)
    plant_p_upper_limits = np.nan_to_num(plant.GetPositionUpperLimits(), posinf=0)

    print(plant_p_lower_limits, plant_p_upper_limits)

    trajopt.AddPositionBounds(plant_p_lower_limits, plant_p_upper_limits)

    plant_v_lower_limits = np.zeros((num_q,))
    plant_v_upper_limits = np.zeros((num_q,))
    plant_v_lower_limits_ = np.nan_to_num(plant.GetVelocityLowerLimits(), neginf=0)
    plant_v_upper_limits_ = np.nan_to_num(plant.GetVelocityUpperLimits(), posinf=0)
    plant_v_lower_limits[:len(plant_v_lower_limits_)] = plant_v_lower_limits_
    plant_v_upper_limits[:len(plant_v_upper_limits_)] = plant_v_upper_limits_

    trajopt.AddVelocityBounds(plant_v_lower_limits, plant_v_upper_limits)

    trajopt.AddDurationConstraint(1, 3)
    start_lim = 1e-2
    end_lim   = 0.2

    constrain_position(plant, trajopt, X_WGStart, 0, plant_temp_context,
                       with_orientation=True, pos_limit=start_lim)
    constrain_position(plant, trajopt, X_WGgoal,  1, plant_temp_context,
                       with_orientation=True, pos_limit=end_lim)

    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 1)

    result = Solve(prog)
    return handle_opt_result(result, trajopt, prog)


def create_open_loop_controller(builder, plant, iiwa, X_G, times, plant_temp_context):
    q_trajectory = solve_for_iiwa_internal_trajectory(plant, X_G, times, plant_temp_context)
    q_trajectory_system = builder.AddSystem(TrajectorySource(q_trajectory))

    num_iiwa_positions = plant.num_positions(iiwa)
    remaining_dim = plant.num_positions() - num_iiwa_positions
    demux = builder.AddSystem(Demultiplexer(output_ports_sizes=[num_iiwa_positions, remaining_dim]))
    builder.Connect(q_trajectory_system.get_output_port(), demux.get_input_port())

    # wsg placeholder
    opened = np.array([0.107]);
    traj_wsg = PiecewisePolynomial.FirstOrderHold([times['initial'], q_trajectory.end_time()],
        np.hstack([[opened], [opened]]))
    wsg_source = builder.AddSystem(TrajectorySource(traj_wsg))

    return demux.get_output_port(0), wsg_source.get_output_port()
