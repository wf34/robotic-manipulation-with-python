#!/usr/bin/env python3

import os
import sys

from typing import Literal

import numpy as np
from tap import Tap

from pydrake.geometry import Meshcat

TIME_STEP=0.007  #faster
IIWA_DEFAULT_POSITION = [-1.57, 0.1, 0, -1.2, 0, 1.6, 0]

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    MultibodyPlant,
    Simulator,
    #ContactVisualizer,
    #ContactVisualizerParams,
    #StateInterpolatorWithDiscreteDerivative,
    #SchunkWsgPositionController,
    #MakeMultibodyStateToWsgStateSystem,

    Parser,
    FindResourceOrThrow,
    RevoluteJoint,
    RigidTransform,
    RollPitchYaw,
)

def AddWsg(plant,
           iiwa_model_instance,
           roll=np.pi / 2.0,
           welded=False,
           sphere=False):
    parser = Parser(plant)
    if welded:
        if sphere:
            gripper = parser.AddModelFromFile(
                FindResource("models/schunk_wsg_50_welded_fingers_sphere.sdf"),
                "gripper")
        else:
            gripper = parser.AddModelFromFile(
                FindResource("models/schunk_wsg_50_welded_fingers.sdf"),
                "gripper")
    else:
        gripper = parser.AddModelFromFile(
            FindResourceOrThrow(
                "drake/manipulation/models/"
                "wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"))

    X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, np.pi / 2.0), [0, 0, 0.114])
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", iiwa_model_instance),
                     plant.GetFrameByName("body", gripper), X_7G)
    return gripper


def AddIiwa(plant, collision_model="no_collision"):
    sdf_path = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/iiwa7/"
        f"iiwa7_{collision_model}.sdf")

    parser = Parser(plant)
    iiwa = parser.AddModelFromFile(sdf_path)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Set default positions:
    q0 = IIWA_DEFAULT_POSITION
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return iiwa

def get_manipuland_resource_path(resource_name: str) -> str:
    resource_file = os.path.join('resources', f'{resource_name}.sdf')
    full_path = os.path.join(os.path.split(os.path.abspath(os.path.dirname(sys.argv[0])))[0], resource_file)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise Exception(f'a resource {resource_name} is absent')
    return full_path

def run_manipulation(method: str):
    meshcat = Meshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=TIME_STEP)
    iiwa = AddIiwa(plant, collision_model="with_box_collision")
    wsg = AddWsg(plant, iiwa, welded=False, sphere=False)

    web_url = meshcat.web_url()
    print(f'Meshcat is now available at {web_url}')
    os.system(f'xdg-open {web_url}')

    if not simulator:
        return

    print(total_time)
    visualizer.StartRecording(False)
    simulator.AdvanceTo(total_time)

class ManipulationArgs(Tap):
    method: Literal['inv-kin', 'global']  # Which controller to use

    def configure(self):
        self.add_argument('-m', '--method')


if __name__ == '__main__':
    print(get_manipuland_resource_path('foam_brick'))
    exit(0)
    run_manipulation(**ManipulationArgs().parse_args().as_dict())
