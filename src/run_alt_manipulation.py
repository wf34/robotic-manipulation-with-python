#!/usr/bin/env python3

import argparse
import functools
import itertools
import time
import typing
import os
import sys

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.all import (AddMultibodyPlantSceneGraph, BsplineTrajectory, Trajectory, GeometryInstance,
                         DiagramBuilder, KinematicTrajectoryOptimization, LinearConstraint,
                         MeshcatVisualizer, MeshcatVisualizerParams, PointToPointDistanceConstraint,
                         Parser, PositionConstraint, OrientationConstraint,
                         Rgba, RigidTransform, Role, Solve, Sphere, PiecewisePolynomial,
                         Meshcat, FindResourceOrThrow, RevoluteJoint, RollPitchYaw, GetDrakePath, MeshcatCone,
                         ConstantVectorSource)
from pydrake.geometry import (Cylinder, GeometryInstance,
                              MakePhongIllustrationProperties)
from pydrake.multibody import inverse_kinematics

from grading import get_start_and_end_positions
from resource_loader import AddIiwa, AddWsg, get_resource, BRICK_GOAL_TRANSLATION, IIWA_DEFAULT_POSITION
from visualization_tools import AddMeshactProgressSphere, AddMeshcatSphere, AddMeshcatTriad

TIME_STEP=0.007  #faster
POSEPS = np.array([1e-2] * 7 + [0.] * 2)

def get_shelf_offset():
    return np.array([-.15, 0.15, 0.13115])

def PublishPositionTrajectores(trajectories, # for iiwa
                               wsg_trajectory,
                               root_context,
                               plant,
                               visualizer,
                               meshcat=None,
                               time_step=1.0 / 33.0):
    if 0 == len(trajectories):
        return

    plant_context = plant.GetMyContextFromRoot(root_context)
    visualizer_context = visualizer.GetMyContextFromRoot(root_context)
    visualizer.StartRecording(False)

    trajectory_end_time = lambda t: 2.0 if not t else t.end_time()
    ends = list(itertools.accumulate(map(trajectory_end_time, trajectories)))
    overall_length = functools.reduce(lambda x, y: x + y, ends, 0)
    begins = [0.] + ends[:-1]

    def get_trajectory_value(current_traj_index, current_time):
        trajectory = trajectories[current_traj_index]
        if not trajectory:
            assert current_traj_index != 0
            prev = trajectories[current_traj_index-1]
            return prev.value(prev.end_time())
        else:
            trajectory = trajectories[current_traj_index]
            return trajectory.value(current_time)

    current_traj_index = 0
    for t in np.append(np.arange(0., overall_length, time_step), overall_length):
        current_time = t - begins[current_traj_index]

        if current_time > trajectory_end_time(trajectories[current_traj_index]):
            current_traj_index += 1

        if current_traj_index == len(trajectories):
            visualizer.ForcedPublish(visualizer_context)
            break

        x = get_trajectory_value(current_traj_index, current_time)
        wsg_current_value = wsg_trajectory.value(t).ravel()[0]
        x[7,0] = wsg_current_value
        plant.SetPositions(plant_context, x)
        if meshcat:
            AddMeshactProgressSphere(meshcat, t, current_traj_index, plant, root_context)
        visualizer.ForcedPublish(visualizer_context)

    visualizer.StopRecording()
    visualizer.PublishRecording()


def create_diagram_without_controllers(meshcat = None):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=TIME_STEP)
    iiwa = AddIiwa(plant, collision_model="with_box_collision")
    wsg = AddWsg(plant, iiwa, welded=False, sphere=False)

    shelves = get_resource(plant, 'shelves')
    X_WS = RigidTransform(RotationMatrix.Identity(), [0.7, 0., 0.4085])
    plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName('shelves_body', shelves),
            X_WS)

    brick_model = get_resource(plant, 'foam_brick')
    brick_body = plant.GetBodyByName('base_link', brick_model)
    X_WB = RigidTransform(RotationMatrix.Identity(), [0.6, 0., 0.4085 + 0.143])
    plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName('base_link', brick_model),
            X_WB)
    plant.Finalize()

    visualizer = None
    if meshcat:
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration))

    diagram = builder.Build()
    diagram.set_name(sys.argv[0] + '/without_controllers')
    return diagram, scene_graph, plant, visualizer


def handle_opt_result(result, trajopt, prog, goal_name):
    if not result.is_success():
        print(dir(result))
        print(result.get_solver_id().name(), result.GetInfeasibleConstraintNames(prog), result.GetInfeasibleConstraints(prog))
        assert False, "Trajectory optimization towards {} has failed".format(goal_name)
    else:
        r = trajopt.ReconstructTrajectory(result)
        print("Trajectory optimization towards {} has succeeded, on interval <{:.2f}, {:.2f}>".format(goal_name, r.start_time(), r.end_time()))
        return r


def get_present_plant_position_with_inf(plant, plant_context, information=1., model_name='iiwa7'):
    iiwa_model_instance = plant.GetModelInstanceByName(model_name)
    indices = list(map(int, plant.GetJointIndices(model_instance=iiwa_model_instance)))
    n = plant.num_positions()
    plant_0 = plant.GetPositions(plant_context)
    plant_inf = np.eye(n) * 1.e-9
    for i in zip(indices):
        plant_inf[i, i] = information
    return plant_0, plant_inf


def get_torque_coords(plant, X_WGgoal, q0, upperbound_m, upperbound_deg):
    lowerbound_m = np.array(upperbound_m) * -1.
    ik = inverse_kinematics.InverseKinematics(plant)
    q_variables = ik.q()
    prog = ik.prog()
    prog.SetInitialGuess(q_variables, q0)
    prog.AddCost(np.square(np.dot(q_variables, q0)))
    ik.AddPositionConstraint(
        frameA=plant.GetFrameByName("body"),
        frameB=plant.world_frame(),
        p_BQ=X_WGgoal.translation(),
        p_AQ_lower=lowerbound_m,
        p_AQ_upper=upperbound_m)
    ik.AddOrientationConstraint(
        frameAbar=plant.GetFrameByName("body"),
        R_AbarA=X_WGgoal.rotation().inverse(),
        frameBbar=plant.world_frame(),
        R_BbarB=RotationMatrix(),
        theta_bound=np.radians(upperbound_deg))
    result = Solve(prog)
    assert result.is_success()
    return result.GetSolution(q_variables)


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


def make_gripper_frames(X_G, X_O, meshcat: typing.Optional[Meshcat] = None) -> typing.Mapping[str, RigidTransform]:
    # returns `X_G`, a dict of "Keyframe" gripper locations that controller must pass through

    p_GgraspO = [0.00, 0.07, -0.03]
    R_GgraspO = RotationMatrix.MakeZRotation(np.pi/2.0)

    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
    X_OGgrasp = X_GgraspO.inverse()

    X_GgraspGpregrasp = RigidTransform([-0.1, -0.2, 0.0])
    X_GgraspGpregrasp2 = RigidTransform([-0.1, -0.3, -0.1])
    X_GgraspGpregrasp3 = RigidTransform([0.1, -0.2, 0.1])

    X_G['pick'] = X_O['initial'].multiply(X_OGgrasp)
    X_G['prepick'] = X_G['pick'].multiply(X_GgraspGpregrasp)

    X_G['place'] = X_O['goal'].multiply(X_OGgrasp)
    X_G['preplace'] = X_G['pick'].multiply(X_GgraspGpregrasp2)
    X_G['postplace'] = X_G['place'].multiply(X_GgraspGpregrasp3)
    X_G['final'] = X_G['initial']

    if meshcat:
        AddMeshcatTriad(meshcat, 'X_Ginitial', X_PT=X_G['initial'])
        AddMeshcatTriad(meshcat, 'X_Gprepick', X_PT=X_G['prepick'])
        AddMeshcatTriad(meshcat, 'X_Gpick', X_PT=X_G['pick'])
        AddMeshcatTriad(meshcat, 'X_Gpreplace', X_PT=X_G['preplace'])
        AddMeshcatTriad(meshcat, 'X_Gplace', X_PT=X_G['place'])
        AddMeshcatTriad(meshcat, 'X_Gpostplace', X_PT=X_G['postplace'])
        AddMeshcatTriad(meshcat, 'X_Gfinal', X_PT=X_G['final'])

    return X_G


def run_traj_opt_towards_prepick(goal_name, X_WGStart, X_WGgoal, plant, plant_context, scene_graph = None):
    num_q = plant.num_positions()
    num_c = 4

    trajopt = KinematicTrajectoryOptimization(num_q, num_c)
    prog = trajopt.get_mutable_prog()

    q0, inf0 = get_present_plant_position_with_inf(plant, plant_context)
    q_goal = get_torque_coords(plant, X_WGgoal, q0, [0.1]*3, 3)
    prog.AddQuadraticErrorCost(inf0, q0, trajopt.control_points()[:, 0])

    q_guess = np.linspace(q0.reshape((num_q, 1)),
                          q_goal.reshape((num_q, 1)),
                          trajopt.num_control_points()
        )[:, :, 0].T
    trajopt.SetInitialGuess(BsplineTrajectory(trajopt.basis(), q_guess))

    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(2.0)

    plant_p_lower_limits = np.nan_to_num(plant.GetPositionLowerLimits(), neginf=0)
    plant_p_upper_limits = np.nan_to_num(plant.GetPositionUpperLimits(), posinf=0)
    trajopt.AddPositionBounds(plant_p_lower_limits, plant_p_upper_limits)

    plant_v_lower_limits = np.nan_to_num(plant.GetVelocityLowerLimits(), neginf=0) / 3.
    plant_v_upper_limits = np.nan_to_num(plant.GetVelocityUpperLimits(), posinf=0) / 3.
    trajopt.AddVelocityBounds(plant_v_lower_limits, plant_v_upper_limits)

    trajopt.AddDurationConstraint(3, 6)

    start_lim = 1e-2
    end_lim   = 0.1

    constrain_position(plant, trajopt, X_WGStart, 0, plant_context,
                       with_orientation=True, pos_limit=start_lim)
    constrain_position(plant, trajopt, X_WGgoal, 1, plant_context,
                       with_orientation=True, pos_limit=end_lim)

    zero_vec = np.zeros((num_q, 1))
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 0)
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 1)

    result = Solve(prog)
    return handle_opt_result(result, trajopt, prog, goal_name)


def run_traj_opt_towards_pick(goal_name, X_WGStart, X_WGgoal, plant, plant_context, scene_graph = None):
    num_q = plant.num_positions()
    num_c = 4

    trajopt = KinematicTrajectoryOptimization(num_q, num_c)
    prog = trajopt.get_mutable_prog()

    q0, inf0 = get_present_plant_position_with_inf(plant, plant_context)
    q_goal = get_torque_coords(plant, X_WGgoal, q0, [0.02, 0.02, 0.02], 3.)

    prog.AddQuadraticErrorCost(inf0, q0, trajopt.control_points()[:, 0])

    q_guess = np.linspace(q0.reshape((num_q, 1)),
                          q_goal.reshape((num_q, 1)),
                          trajopt.num_control_points()
        )[:, :, 0].T
    trajopt.SetInitialGuess(BsplineTrajectory(trajopt.basis(), q_guess))

    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(2.0)

    plant_p_lower_limits = np.nan_to_num(plant.GetPositionLowerLimits(), neginf=0)
    plant_p_upper_limits = np.nan_to_num(plant.GetPositionUpperLimits(), posinf=0)
    trajopt.AddPositionBounds(plant_p_lower_limits, plant_p_upper_limits)

    plant_v_lower_limits = np.nan_to_num(plant.GetVelocityLowerLimits(), neginf=0) / 6.
    plant_v_upper_limits = np.nan_to_num(plant.GetVelocityUpperLimits(), posinf=0) / 6.

    #accel_bounds_vec = np.array([0.05]*7+[0.]*2)
    #trajopt.AddAccelerationBounds(-accel_bounds_vec, accel_bounds_vec)


    trajopt.AddPathPositionConstraint(q0 - POSEPS, q0 + POSEPS, 1e-3)

    trajopt.AddDurationConstraint(2, 7)

    start_lim = 1e-2
    end_lim   = 0.01

    constrain_position(plant, trajopt, X_WGStart, 0, plant_context,
                       with_orientation=True, pos_limit=start_lim)
    constrain_position(plant, trajopt, X_WGgoal, 1, plant_context,
                       with_orientation=True, pos_limit=end_lim)

    zero_vec = np.zeros((num_q, 1))
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 0)
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 1)

    result = Solve(prog)
    return handle_opt_result(result, trajopt, prog, goal_name)


def run_traj_opt_towards_preplace(goal_name, X_WGStart, X_WGgoal, plant, plant_context, scene_graph):
    num_q = plant.num_positions()
    num_c = 7

    trajopt = KinematicTrajectoryOptimization(num_q, num_c)
    prog = trajopt.get_mutable_prog()

    q0, inf0 = get_present_plant_position_with_inf(plant, plant_context)
    q_goal = get_torque_coords(plant, X_WGgoal, q0, [0.04, 0.04, 0.7], 3.)

    prog.AddQuadraticErrorCost(inf0, q0, trajopt.control_points()[:, 0])
    q_guess = np.linspace(q0.reshape((num_q, 1)),
                          q_goal.reshape((num_q, 1)),
                          trajopt.num_control_points()
        )[:, :, 0].T
    trajopt.SetInitialGuess(BsplineTrajectory(trajopt.basis(), q_guess))



    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(2.0)

    plant_p_lower_limits = np.nan_to_num(plant.GetPositionLowerLimits(), neginf=0)
    plant_p_upper_limits = np.nan_to_num(plant.GetPositionUpperLimits(), posinf=0)
    trajopt.AddPositionBounds(plant_p_lower_limits, plant_p_upper_limits)

    plant_v_lower_limits = np.nan_to_num(plant.GetVelocityLowerLimits(), neginf=0) / 3.
    plant_v_upper_limits = np.nan_to_num(plant.GetVelocityUpperLimits(), posinf=0) / 3.
    trajopt.AddVelocityBounds(plant_v_lower_limits, plant_v_upper_limits)

    trajopt.AddPathPositionConstraint(q0 - POSEPS, q0 + POSEPS, 1e-3)
    trajopt.AddDurationConstraint(2, 6)

    start_lim = 1e-2
    end_lim   = 0.12

    constrain_position(plant, trajopt, X_WGStart, 0, plant_context,
                       with_orientation=True, pos_limit=start_lim)
    constrain_position(plant, trajopt, X_WGgoal, 1, plant_context,
                       with_orientation=True, pos_limit=end_lim, theta_bound_degrees=15.)

    zero_vec = np.zeros((num_q, 1))
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 0)
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 1)

    result = Solve(prog)
    return handle_opt_result(result, trajopt, prog, goal_name)


def run_traj_opt_towards_place(goal_name, X_WGStart, X_WGgoal, plant, plant_context, scene_graph):
    num_q = plant.num_positions()
    num_c = 10

    trajopt = KinematicTrajectoryOptimization(num_q, num_c)
    prog = trajopt.get_mutable_prog()

    q0, inf0 = get_present_plant_position_with_inf(plant, plant_context, information=100)
    q_goal = get_torque_coords(plant, X_WGgoal, q0, [0.1, 0.1, 0.1], 3.)
    prog.AddQuadraticErrorCost(inf0, q0, trajopt.control_points()[:, 0])

    q_guess = np.linspace(q0.reshape((num_q, 1)),
                          q_goal.reshape((num_q, 1)),
                          trajopt.num_control_points()
        )[:, :, 0].T
    trajopt.SetInitialGuess(BsplineTrajectory(trajopt.basis(), q_guess))

    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(2.0)

    plant_p_lower_limits = np.nan_to_num(plant.GetPositionLowerLimits(), neginf=0)
    plant_p_upper_limits = np.nan_to_num(plant.GetPositionUpperLimits(), posinf=0)
    trajopt.AddPositionBounds(plant_p_lower_limits, plant_p_upper_limits)

    plant_v_lower_limits = np.nan_to_num(plant.GetVelocityLowerLimits(), neginf=0) / 3.
    plant_v_upper_limits = np.nan_to_num(plant.GetVelocityUpperLimits(), posinf=0) / 3.
    trajopt.AddVelocityBounds(plant_v_lower_limits, plant_v_upper_limits)

    trajopt.AddPathPositionConstraint(q0 - POSEPS, q0 + POSEPS, 1e-4)
    trajopt.AddDurationConstraint(2, 4)

    finger_frame = plant.GetBodyByName('body').body_frame()
    shelf_frame = plant.GetBodyByName('shelves_body').body_frame()

    another_dist_c = PointToPointDistanceConstraint(plant,
                                                    finger_frame, np.array([0,0,0]),
                                                    shelf_frame, get_shelf_offset(),
                                                    0.1, 100, plant_context)
    trajopt.AddPathPositionConstraint(another_dist_c, 0.5)

    start_lim = 1e-2
    end_lim   = 0.05

    constrain_position(plant, trajopt, X_WGStart, 0, plant_context,
                       with_orientation=True, pos_limit=start_lim)
    constrain_position(plant, trajopt, X_WGgoal, 1, plant_context,
                       with_orientation=True, pos_limit=end_lim, theta_bound_degrees=15.)

    zero_vec = np.zeros((num_q, 1))
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 0)
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 1)

    result = Solve(prog)
    return handle_opt_result(result, trajopt, prog, goal_name)


def run_traj_opt_towards_postplace(goal_name, X_WGStart, X_WGgoal, plant, plant_context, scene_graph = None):
    num_q = plant.num_positions()
    num_c = 5

    trajopt = KinematicTrajectoryOptimization(num_q, num_c)
    prog = trajopt.get_mutable_prog()

    q0, inf0 = get_present_plant_position_with_inf(plant, plant_context)
    q_goal = get_torque_coords(plant, X_WGgoal, q0, [0.2, 0.2, 0.2], 3.)
    prog.AddQuadraticErrorCost(inf0, q0, trajopt.control_points()[:, 0])

    q_guess = np.linspace(q0.reshape((num_q, 1)),
                          q_goal.reshape((num_q, 1)),
                          trajopt.num_control_points()
        )[:, :, 0].T
    trajopt.SetInitialGuess(BsplineTrajectory(trajopt.basis(), q_guess))

    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(2.0)

    plant_p_lower_limits = np.nan_to_num(plant.GetPositionLowerLimits(), neginf=0)
    plant_p_upper_limits = np.nan_to_num(plant.GetPositionUpperLimits(), posinf=0)
    trajopt.AddPositionBounds(plant_p_lower_limits, plant_p_upper_limits)

    plant_v_lower_limits = np.nan_to_num(plant.GetVelocityLowerLimits(), neginf=0) / 3.
    plant_v_upper_limits = np.nan_to_num(plant.GetVelocityUpperLimits(), posinf=0) / 3.
    trajopt.AddVelocityBounds(plant_v_lower_limits, plant_v_upper_limits)

    trajopt.AddPathPositionConstraint(q0 - POSEPS, q0 + POSEPS, 1e-4)
    trajopt.AddDurationConstraint(1, 4)

    start_lim = 1e-2
    end_lim   = 0.1

    constrain_position(plant, trajopt, X_WGStart, 0, plant_context,
                       with_orientation=True, pos_limit=start_lim)
    constrain_position(plant, trajopt, X_WGgoal, 1, plant_context,
                       with_orientation=False, pos_limit=end_lim)

    zero_vec = np.zeros((num_q, 1))
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 0)
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 1)

    result = Solve(prog)
    return handle_opt_result(result, trajopt, prog, goal_name)


def run_traj_opt_towards_final(goal_name, X_WGStart, X_WGgoal, plant, plant_context, scene_graph = None):
    num_q = plant.num_positions()
    num_c = 5

    trajopt = KinematicTrajectoryOptimization(num_q, num_c)
    prog = trajopt.get_mutable_prog()

    q0, inf0 = get_present_plant_position_with_inf(plant, plant_context, information=100)
    q_goal = get_torque_coords(plant, X_WGgoal, q0, [0.05, 0.05, 0.05], 3.)
    q_guess = np.linspace(q0.reshape((num_q, 1)),
                          q_goal.reshape((num_q, 1)),
                          trajopt.num_control_points()
        )[:, :, 0].T
    trajopt.SetInitialGuess(BsplineTrajectory(trajopt.basis(), q_guess))

    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(2.0)

    plant_p_lower_limits = np.nan_to_num(plant.GetPositionLowerLimits(), neginf=0)
    plant_p_upper_limits = np.nan_to_num(plant.GetPositionUpperLimits(), posinf=0)
    trajopt.AddPositionBounds(plant_p_lower_limits, plant_p_upper_limits)

    plant_v_lower_limits = np.nan_to_num(plant.GetVelocityLowerLimits(), neginf=0) / 3.
    plant_v_upper_limits = np.nan_to_num(plant.GetVelocityUpperLimits(), posinf=0) / 3.
    trajopt.AddVelocityBounds(plant_v_lower_limits, plant_v_upper_limits)

    trajopt.AddPathPositionConstraint(q0 - POSEPS, q0 + POSEPS, 1e-4)
    trajopt.AddDurationConstraint(1, 4)

    start_lim = 1e-2
    end_lim   = 0.1

    constrain_position(plant, trajopt, X_WGStart, 0, plant_context,
                       with_orientation=True, pos_limit=start_lim)
    constrain_position(plant, trajopt, X_WGgoal, 1, plant_context,
                       with_orientation=True, pos_limit=end_lim)

    zero_vec = np.zeros((num_q, 1))
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 0)
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 1)

    result = Solve(prog)
    return handle_opt_result(result, trajopt, prog, goal_name)


def make_wsg_command_trajectory(times) -> Trajectory:
    opened = np.array([0.107]);
    closed = np.array([0.0]);
    traj_wsg = PiecewisePolynomial.FirstOrderHold([times['initial'], times['prepick']],
                                                  np.hstack([[opened], [opened]]))

    for n, v in zip(['pick', 'pick_close', 'place', 'place_open', 'postplace', 'final'],
                    [opened, closed,       closed,  opened,        opened,     opened]):
        if n in times:
            traj_wsg.AppendFirstOrderSegment(times[n], v)

    return traj_wsg


def solve_for_picking_trajectories(scene_graph, plant, X_G, X_O, plant_context, meshcat) -> typing.Tuple[typing.List[typing.Optional[BsplineTrajectory]], Trajectory]:
    # returns a tuple (stacked_iiwa_trajectores, wsg_trajectory)
    # stacked_iiwa_trajectores are the manipulator trajectories, found by the `KinematicTrajectoryOptimization`
    # `None` entry in a `stacked_iiwa_trajectores` signifies stationary trajectory;
    # stationary trajectories are used to give the controller time to close the gripper

    gripper_body_index = int(plant.GetBodyByName('body').index())
    X_WBcurrent_getter = lambda body_index: plant.get_body_poses_output_port().Eval(plant_context)[body_index]
    X_WGcurrent_getter = lambda _=None: X_WBcurrent_getter(gripper_body_index)

    X_G = make_gripper_frames(X_G, X_O, meshcat)
    goal_frames = ['prepick', 'pick', 'preplace', 'place', 'postplace', 'final']

    funcs = {
        'prepick': run_traj_opt_towards_prepick,
        'pick': run_traj_opt_towards_pick,
        'preplace': run_traj_opt_towards_preplace,
        'place': run_traj_opt_towards_place,
        'postplace': run_traj_opt_towards_postplace,
        'final': run_traj_opt_towards_final,
    }

    stacked_trajectores = []
    for goal_name in goal_frames:
        X_WGStart = X_WGcurrent_getter()
        X_WGgoal = X_G[goal_name]

        if goal_name not in funcs:
            break
        interm_traj = funcs[goal_name](goal_name, X_WGStart, X_WGgoal, plant, plant_context, scene_graph)
        if not interm_traj:
            break

        plant.SetPositions(plant_context, interm_traj.FinalValue())
        stacked_trajectores.append(interm_traj)
        if goal_name in ('pick', 'place'):
            stacked_trajectores.append(None) # placeholder trajectory

    print('produced stack', len(stacked_trajectores))
    start_frames_with_placeholders = ['initial', 'prepick', 'pick', 'pick_close', 'preplace', 'place', 'place_open', 'postplace', 'final']
    ends = list(itertools.accumulate(map(lambda x: 2.0 if not x else x.end_time(), stacked_trajectores)))
    times = {}
    for a, b in zip(start_frames_with_placeholders, [0.] + ends):
        times[a] = b
    wsg_trajectory = make_wsg_command_trajectory(times)

    prev_traj = None
    for i, t in enumerate(stacked_trajectores[:-1]):
        if t is None:
            continue
        if 0 != i:
            beg_vec = t.value(t.start_time()).T
            print(i, 'starts with', beg_vec, 'having loss at ', np.linalg.norm(prev_traj - beg_vec))

        prev_traj = t.value(t.end_time()).T
        print(i, '  ends with', prev_traj)
    return stacked_trajectores, wsg_trajectory

def solve_for_iiwa_internal_trajectory_standalone(X_G, X_O):
    diagram, scene_graph, plant, visualizer_absent = create_diagram_without_controllers()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    traj_dimensionality = plant.num_positions()

    return *solve_for_picking_trajectories(scene_graph, plant, X_G, X_O, plant_context, None), traj_dimensionality


def trajopt_demo(meshcat):
    diagram, scene_graph, plant, visualizer = create_diagram_without_controllers(meshcat)
    AddMeshcatSphere(meshcat, 'goal-meshcat', BRICK_GOAL_TRANSLATION)
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    visualizer_context = visualizer.GetMyContextFromRoot(context)
    X_G, X_O = get_start_and_end_positions(plant, plant_context)

    iiwa_trajectories, wsg_trajectory = solve_for_picking_trajectories(scene_graph, plant, X_G, X_O, plant_context, meshcat)
    PublishPositionTrajectores(iiwa_trajectories, wsg_trajectory, context, plant, visualizer, meshcat)


def run_alt_main():
    meshcat = Meshcat()
    web_url = meshcat.web_url()
    #os.system(f'xdg-open {web_url}')
    os.popen(f'chromium {web_url}')
    trajopt_demo(meshcat)
    print('python sent to sleep')
    time.sleep(30)


if '__main__' == __name__:
    np.set_printoptions(linewidth=160)
    run_alt_main()
