import typing
import numpy as np

from pydrake.all import (
    Cylinder,
    Sphere,
    Rgba,

    RotationMatrix,
    RigidTransform,
    Meshcat
)

def AddMeshcatTriad(meshcat, path, length=0.25, radius=0.01, opacity=1.0, X_PT=RigidTransform()):
    meshcat.SetTransform(path, X_PT)
    # x-axis
    X_TG = RigidTransform(
        RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0]
    )
    meshcat.SetTransform(path + '/x-axis', X_TG)
    meshcat.SetObject(
        path + '/x-axis', Cylinder(radius, length), Rgba(1, 0, 0, opacity)
    )

    # y-axis
    X_TG = RigidTransform(
        RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0]
    )
    meshcat.SetTransform(path + '/y-axis', X_TG)
    meshcat.SetObject(
        path + '/y-axis', Cylinder(radius, length), Rgba(0, 1, 0, opacity)
    )

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    meshcat.SetTransform(path + '/z-axis', X_TG)
    meshcat.SetObject(
        path + '/z-axis', Cylinder(radius, length), Rgba(0, 0, 1, opacity)
    )


def AddMeshcatSphere(meshcat: Meshcat, path: str, translation: typing.List[float]):
    radius = 0.025
    X_PT = RigidTransform(RotationMatrix.Identity(), translation)
    meshcat.SetTransform(path, X_PT)
    meshcat.SetObject(
        path, Sphere(radius), Rgba(0, 1, 0, 0.25)
    )


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
