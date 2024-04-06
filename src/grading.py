import copy

import numpy as np

from pydrake.all import MultibodyPlant, Context
from resource_loader import BRICK_GOAL_TRANSLATION

def verify_frames(X_G, times):
    locations = set(X_G.keys())
    times = set(times.keys())
    return locations == times


def get_start_and_end_positions(plant: MultibodyPlant, context: Context):
    # gripper and object positions
    X_G = {'initial': plant.EvalBodyPoseInWorld(context, plant.GetBodyByName('body'))}
    X_O = {'initial': plant.EvalBodyPoseInWorld(context, plant.GetBodyByName('base_link'))}

    X_O['goal'] = copy.deepcopy(X_O['initial'])
    X_O['goal'].set_translation(BRICK_GOAL_TRANSLATION)
    return X_G, X_O


def print_score(plant: MultibodyPlant, context: Context):
    plant_context = plant.GetMyContextFromRoot(context)
    p_WB = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName('base_link')).translation()
    dst = np.linalg.norm(p_WB - BRICK_GOAL_TRANSLATION)
    print('Distance between the brick and its target: {:.3f}; time taken {:.3f}'.format(dst, plant_context.get_time()))
