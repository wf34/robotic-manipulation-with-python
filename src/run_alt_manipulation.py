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
from pydrake.all import (AddMultibodyPlantSceneGraph, BsplineTrajectory, GeometryInstance,
                         DiagramBuilder, KinematicTrajectoryOptimization, LinearConstraint,
                         MeshcatVisualizer, MeshcatVisualizerParams,
                         MinimumDistanceConstraint, Parser, PositionConstraint, OrientationConstraint,
                         Rgba, RigidTransform, Role, Solve, Sphere, PiecewisePolynomial,
                         Meshcat, FindResourceOrThrow, RevoluteJoint, RollPitchYaw, GetDrakePath, MeshcatCone,
                         ConstantVectorSource, StackedTrajectory)
from pydrake.geometry import (Cylinder, GeometryInstance,
                              MakePhongIllustrationProperties)
from pydrake.multibody import inverse_kinematics

from grading import get_start_and_end_positions
from resource_loader import AddIiwa, AddWsg, get_resource, BRICK_GOAL_TRANSLATION, IIWA_DEFAULT_POSITION
from visualization_tools import AddMeshactProgressSphere, AddMeshcatSphere, AddMeshcatTriad

TIME_STEP=0.007  #faster


def PublishPositionTrajectores(trajectory,
                               root_context,
                               plant,
                               visualizer,
                               meshcat=None,
                               time_step=1.0 / 33.0):
    plant_context = plant.GetMyContextFromRoot(root_context)
    visualizer_context = visualizer.GetMyContextFromRoot(root_context)
    visualizer.StartRecording(False)

    overall_length = trajectory.end_time()
    for t in np.append(np.arange(0., overall_length, time_step), overall_length):
        x = trajectory.value(t)
        plant.SetPositions(plant_context, x)
        if meshcat:
            AddMeshactProgressSphere(meshcat, t, overall_length, plant, root_context)
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

    brick_model = get_resource(plant, 'shadow_brick')
    brick_body = plant.GetBodyByName('base_link', brick_model)
    X_WB = RigidTransform(RotationMatrix.Identity(), [0.6, 0., 0.4085 + 0.15])
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
    return diagram, plant, visualizer


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


def make_gripper_frames(X_G, X_O, meshcat: typing.Optional[Meshcat] = None):
    p_GgraspO = [0.05, 0.07, 0.]
    R_GgraspO = RotationMatrix.MakeZRotation(np.pi/2.0)

    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
    X_OGgrasp = X_GgraspO.inverse()

    X_GgraspGpregrasp = RigidTransform([0, -0.2, 0.0])

    X_G['pick'] = X_O['initial'].multiply(X_OGgrasp)
    X_G['prepick'] = X_G['pick'].multiply(X_GgraspGpregrasp)

    if meshcat:
        AddMeshcatTriad(meshcat, 'X_Ginitial', X_PT=X_G['initial'])
        AddMeshcatTriad(meshcat, 'X_Gprepick', X_PT=X_G['prepick'])
        AddMeshcatTriad(meshcat, 'X_Gpick', X_PT=X_G['pick'])

    return X_G

def solve_for_iiwa_internal_trajectory(plant, X_G, X_O, plant_temp_context, meshcat :typing.Optional[Meshcat] = None):
    num_q = plant.num_positions()
    num_c = 15
    print('num_positions: {}; num control points: {}'.format(num_q, num_c))
    X_G = make_gripper_frames(X_G, X_O, meshcat)

    X_WGStart = X_G['initial']
    X_WGgoal = X_G['prepick']

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
    trajopt.AddPathLengthCost(2.0)

    plant_p_lower_limits = np.nan_to_num(plant.GetPositionLowerLimits(), neginf=0)
    plant_p_upper_limits = np.nan_to_num(plant.GetPositionUpperLimits(), posinf=0)
    trajopt.AddPositionBounds(plant_p_lower_limits, plant_p_upper_limits)

    plant_v_lower_limits = np.nan_to_num(plant.GetVelocityLowerLimits(), neginf=0)
    plant_v_upper_limits = np.nan_to_num(plant.GetVelocityUpperLimits(), posinf=0)
    trajopt.AddVelocityBounds(plant_v_lower_limits, plant_v_upper_limits)

    trajopt.AddDurationConstraint(3, 5)

    start_lim = 1e-2
    end_lim   = 0.1

    constrain_position(plant, trajopt, X_WGStart, 0, plant_temp_context,
                       with_orientation=True, pos_limit=start_lim)
    constrain_position(plant, trajopt, X_WGgoal, 1, plant_temp_context,
                       with_orientation=True, pos_limit=end_lim)

    zero_vec = np.zeros((num_q, 1))
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 0)
    trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 1)

    result = Solve(prog)
    return handle_opt_result(result, trajopt, prog)


def solve_for_iiwa_internal_trajectory_standalone(X_G, X_O):
    diagram, plant, visualizer_absent = create_diagram_without_controllers()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    traj_dimensionality = plant.num_positions()

    return solve_for_iiwa_internal_trajectory(plant, X_G, X_O, plant_context, None), traj_dimensionality


def trajopt_demo(meshcat):
    diagram, plant, visualizer = create_diagram_without_controllers(meshcat)
    AddMeshcatSphere(meshcat, 'goal-meshcat', BRICK_GOAL_TRANSLATION)
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    visualizer_context = visualizer.GetMyContextFromRoot(context)
    X_G, X_O = get_start_and_end_positions(plant, plant_context)

    trajectory = solve_for_iiwa_internal_trajectory(plant, X_G, X_O, plant_context, meshcat)

    print(trajectory.start_time(), trajectory.end_time())
    PublishPositionTrajectores(trajectory, context, plant, visualizer, meshcat)


def run_alt_main():
    meshcat = Meshcat()
    web_url = meshcat.web_url()
    os.system(f'xdg-open {web_url}')
    trajopt_demo(meshcat)
    print('python sent to sleep')
    time.sleep(30)


if '__main__' == __name__:
    run_alt_main()
