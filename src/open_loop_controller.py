
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



from run_alt_manipulation import solve_for_iiwa_internal_trajectory_standalone


def create_open_loop_controller(builder, plant, iiwa, X_G, times):
    q_trajectory, trajectory_dim = solve_for_iiwa_internal_trajectory_standalone(X_G, times)
    q_trajectory_system = builder.AddSystem(TrajectorySource(q_trajectory))

    num_iiwa_positions = plant.num_positions(iiwa)
    remaining_dim = trajectory_dim - num_iiwa_positions
    demux = builder.AddSystem(Demultiplexer(output_ports_sizes=[num_iiwa_positions, remaining_dim]))
    builder.Connect(q_trajectory_system.get_output_port(), demux.get_input_port())

    # wsg placeholder
    opened = np.array([0.107]);
    traj_wsg = PiecewisePolynomial.FirstOrderHold([times['initial'], q_trajectory.end_time()],
        np.hstack([[opened], [opened]]))
    wsg_source = builder.AddSystem(TrajectorySource(traj_wsg))

    return demux.get_output_port(0), wsg_source.get_output_port()
