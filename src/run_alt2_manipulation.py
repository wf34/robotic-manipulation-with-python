#!/usr/bin/env python3
import time
import os
import numpy as np

from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.all import (AddMultibodyPlantSceneGraph, BsplineTrajectory, Trajectory, GeometryInstance,
                         DiagramBuilder, KinematicTrajectoryOptimization, LinearConstraint,
                         MeshcatVisualizer, MeshcatVisualizerParams,
                         Parser, PositionConstraint, OrientationConstraint, DistanceConstraint,
                         Rgba, RigidTransform, Role, Solve, Sphere, PiecewisePolynomial,
                         Meshcat, FindResourceOrThrow, RevoluteJoint, RollPitchYaw, GetDrakePath, MeshcatCone,
                         ConstantVectorSource)

from differential_controller import make_gripper_frames, make_gripper_trajectory, make_wsg_command_trajectory
from run_alt_manipulation import create_diagram_without_controllers, get_torque_coords, get_present_plant_position_with_inf

from resource_loader import AddIiwa, AddWsg, get_resource, BRICK_GOAL_TRANSLATION, IIWA_DEFAULT_POSITION
from visualization_tools import AddMeshcatSphere, AddMeshactProgressSphere

from grading import get_start_and_end_positions

def PublishPositionTrajectory(trajectory, # for iiwa
                              wsg_trajectory,
                              root_context,
                              plant,
                              visualizer,
                              meshcat=None,
                              times=None,
                              time_step=1./5.):
    plant_context = plant.GetMyContextFromRoot(root_context)
    visualizer_context = visualizer.GetMyContextFromRoot(root_context)
    visualizer.StartRecording(False)

    trajectory_end_time = trajectory.end_time()
    overall_length = trajectory_end_time

    prev_q, _ = None, None #get_present_plant_position_with_inf(plant, plant_context)
    n = plant.num_positions()

    def get_index(t):
        if times is None:
            return 0

        print(times)
        for i, time in enumerate(times.values()):
            if t <= time:
                return i
        return len(times)


    for t in np.arange(0., overall_length, time_step):
        x_wg = RigidTransform(trajectory.value(t))
        i = get_index(t)
        #q_goal = get_torque_coords(plant, x_wg, prev_q, [0.6, 0.6, 0.6])
        #wsg_current_value = wsg_trajectory.value(t).ravel()[0]
        #q_goal[7] = wsg_current_value
        #plant.SetPositions(plant_context, q_goal)
        if meshcat:
            AddMeshactProgressSphere(meshcat, t, i, plant, root_context, x_wg.translation())
        visualizer.ForcedPublish(visualizer_context)
        #prev_q = q_goal

    visualizer.ForcedPublish(visualizer_context)
    visualizer.StopRecording()
    visualizer.PublishRecording()

def invkin_demo(meshcat):
    diagram, scene_graph, plant, visualizer = create_diagram_without_controllers(meshcat)
    AddMeshcatSphere(meshcat, 'goal-meshcat', BRICK_GOAL_TRANSLATION)
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    visualizer_context = visualizer.GetMyContextFromRoot(context)
    X_G, X_O = get_start_and_end_positions(plant, plant_context)
    X_G, times = make_gripper_frames(X_G, X_O)

    traj = make_gripper_trajectory(X_G, times)
    traj_wsg_command = make_wsg_command_trajectory(times)
    PublishPositionTrajectory(traj, traj_wsg_command, context, plant, visualizer, meshcat, times)


def run_alt_main():
    meshcat = Meshcat()
    web_url = meshcat.web_url()
    #os.system(f'xdg-open {web_url}')
    os.popen(f'chromium {web_url}')
    invkin_demo(meshcat)
    print('python sent to sleep')
    time.sleep(30)


if '__main__' == __name__:
    run_alt_main()
