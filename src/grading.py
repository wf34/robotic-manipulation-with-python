import copy

from pydrake.all import MultibodyPlant, Context
from resource_loader import BRICK_GOAL_TRANSLATION

def get_start_and_end_positions(plant: MultibodyPlant, context: Context):
    # gripper and object positions
    X_G = {'initial': plant.EvalBodyPoseInWorld(context, plant.GetBodyByName('body'))}
    X_O = {'initial': plant.EvalBodyPoseInWorld(context, plant.GetBodyByName('base_link'))}

    X_O['goal'] = copy.deepcopy(X_O['initial'])
    X_O['goal'].set_translation(BRICK_GOAL_TRANSLATION)
    return X_G, X_O
