#!/usr/bin/env python3

import argparse
import functools
import itertools
import time
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

TIME_STEP=0.007  #faster

from open_loop_controller import solve_for_iiwa_internal_trajectory 
from run_manipulation import AddIiwa, AddWsg, get_resource

def AddMeshactProgressSphere(meshcat, current_time, total_duration, plant, root_context):
    plant_context = plant.GetMyContextFromRoot(root_context)

    blue = np.array([0., 0., 1., 1.])
    green = np.array([0., 1., 0., 1.])

    a = current_time / total_duration
    assert 0. <= a and a <= 1.
    b = 1. - a
    mixture = a * blue + b * green

    root_context.SetTime(current_time)

    X_W_G = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("body"))
    curr_point = 'point_{}'.format(current_time)
    meshcat.SetObject(curr_point, Sphere(0.01), rgba=Rgba(*mixture.tolist()))
    meshcat.SetTransform(curr_point, X_W_G)


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
        print(x.ravel())
        plant.SetPositions(plant_context, x)
        if meshcat:
            AddMeshactProgressSphere(meshcat, t, overall_length, plant, root_context)
        visualizer.ForcedPublish(visualizer_context)

    visualizer.StopRecording()
    visualizer.PublishRecording()


def trajopt_demo(meshcat):
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

    plant.Finalize()

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kIllustration))
    diagram = builder.Build()
    diagram.set_name(sys.argv[0])

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    visualizer_context = visualizer.GetMyContextFromRoot(context)
    times = None
    X_G = {'initial': plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName('body'))}
    trajectory = solve_for_iiwa_internal_trajectory(plant, X_G, times, plant_context)
    print(trajectory.start_time(), trajectory.end_time())
    PublishPositionTrajectores(trajectory, context, plant, visualizer, meshcat)


def run_alt_main():
    meshcat = Meshcat()
    web_url = meshcat.web_url()
    #os.system(f'xdg-open {web_url}')
    trajopt_demo(meshcat)
    exit(0)
    print('python sent to sleep')
    time.sleep(30)


if '__main__' == __name__:
    run_alt_main()
