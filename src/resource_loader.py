
import os
import sys

import numpy as np

from pydrake.all import (
    Parser,
    FindResourceOrThrow,
    RevoluteJoint,
    RigidTransform,
    RollPitchYaw,
    MultibodyPlant,
)

IIWA_DEFAULT_POSITION = [-1.57, 0.1, 0, -1.2, 0, 1.6, 0]
BRICK_GOAL_TRANSLATION = [0.6, 0., 0.2615 + 0.05]

def FindResource(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def AddWsg(plant,
           iiwa_model_instance,
           roll=np.pi / 2.0,
           welded=False,
           sphere=False):
    parser = Parser(plant)
    if welded:
        if sphere:
            gripper = parser.AddModels(
                FindResource('models/schunk_wsg_50_welded_fingers_sphere.sdf'))[0]
        else:
            gripper = parser.AddModels(
                FindResource('models/schunk_wsg_50_welded_fingers.sdf'))[0]
    else:
        gripper = parser.AddModels(
            FindResourceOrThrow(
                'drake/manipulation/models/'
                'wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf'))[0]

    X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, np.pi / 2.0), [0, 0, 0.114])
    plant.WeldFrames(plant.GetFrameByName('iiwa_link_7', iiwa_model_instance),
                     plant.GetFrameByName('body', gripper), X_7G)
    return gripper


def AddIiwa(plant, collision_model='no_collision'):
    sdf_path = FindResourceOrThrow(
        'drake/manipulation/models/iiwa_description/iiwa7/'
        f'iiwa7_{collision_model}.sdf')

    parser = Parser(plant)
    iiwa = parser.AddModels(sdf_path)[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName('iiwa_link_0'))

    # Set default positions:
    q0 = IIWA_DEFAULT_POSITION
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return iiwa


def get_resource_path(resource_name: str) -> str:
    resource_file = os.path.join('resources', f'{resource_name}.sdf')
    full_path = os.path.join(os.path.split(os.path.abspath(os.path.dirname(sys.argv[0])))[0], resource_file)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise Exception(f'a resource {resource_name} is absent')
    return full_path


def get_resource(plant: MultibodyPlant, resource_name: str):
    resource_path = get_resource_path(resource_name)
    return Parser(plant=plant).AddModels(resource_path)[0]
