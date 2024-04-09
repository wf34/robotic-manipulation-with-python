
import itertools
import functools
import numpy as np

from pydrake.all import (
    TrajectorySource, Demultiplexer, LeafSystem,
)

from run_alt_manipulation import solve_for_iiwa_internal_trajectory_standalone

def get_trajectory_end_time(t):
    return 2.0 if not t else t.end_time()


class UnstackTrajectories(LeafSystem):
    def __init__(self, trajectories_stack, trajectory_dim):
        LeafSystem.__init__(self)
        self.DeclareVectorOutputPort("iiwa_inner_coords", trajectory_dim, self.CalcIiwaInnerCoords)
        self.opt_trajectories = trajectories_stack


    def get_entry(self, context):
        target_time = context.get_time()
        opt_trajectories = self.opt_trajectories
        end_times = list(itertools.accumulate(map(get_trajectory_end_time, opt_trajectories)))
        for i, (t, curr_end_time) in enumerate(zip(opt_trajectories, end_times)):
            is_last = (len(opt_trajectories) - 1) == i
            eps = 1.e-6
            assert not is_last or target_time - eps < curr_end_time, 'target_time={:.1f} end_time={:.1f} || trajes ||={} ; cur={}'.format(target_time, curr_end_time, len(opt_trajectories), i)

            if target_time > curr_end_time:
                continue
            else:
                prev_end_time = 0. if i == 0 else end_times[i-1]
                local_target_time = target_time - prev_end_time
                break

        assert 0. <= local_target_time and local_target_time <= curr_end_time, '{:.1f} not in ({:.1f}, {:.1f})'.format(target_time, 0., curr_end_time)

        if t:
            return t.value(local_target_time)
        else:
            assert 0 != i
            prev_traj = opt_trajectories[i-1]
            return prev_traj.value(prev_traj.end_time())

    def CalcIiwaInnerCoords(self, context, output):
        output.set_value(self.get_entry(context))



def create_open_loop_controller(builder, plant, iiwa, X_G, X_O):
    q_trajectories_stack, wsg_trajectory, trajectory_dim = solve_for_iiwa_internal_trajectory_standalone(X_G, X_O)
    total_time = functools.reduce(lambda x, y: x + get_trajectory_end_time(y), q_trajectories_stack, 0.)
    q_trajectory_system = builder.AddSystem(UnstackTrajectories(q_trajectories_stack, trajectory_dim))

    num_iiwa_positions = plant.num_positions(iiwa)
    remaining_dim = trajectory_dim - num_iiwa_positions
    demux = builder.AddSystem(Demultiplexer(output_ports_sizes=[num_iiwa_positions, remaining_dim]))
    builder.Connect(q_trajectory_system.GetOutputPort('iiwa_inner_coords'), demux.get_input_port())

    wsg_source = builder.AddSystem(TrajectorySource(wsg_trajectory))

    return demux.get_output_port(0), wsg_source.get_output_port(), total_time
