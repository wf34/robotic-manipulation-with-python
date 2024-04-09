#!/usr/bin/env python3

import copy
import os
import sys
import time
import typing

import numpy as np
from tap import Tap
import pydot

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Meshcat,
    MeshcatVisualizer,
    MultibodyPlant,
    Simulator,
    StateInterpolatorWithDiscreteDerivative,
    SchunkWsgPositionController,
    MakeMultibodyStateToWsgStateSystem,

    RigidTransform,
    RotationMatrix,

    Adder,
    Demultiplexer,
    PassThrough,
    InverseDynamicsController,

    Cylinder,
    Sphere,
    Rgba,
)

from resource_loader import AddIiwa, AddWsg, get_resource
from differential_controller import create_differential_controller
from open_loop_controller import create_open_loop_controller

TIME_STEP=0.007  #faster


def create_iiwa_controller(plant, iiwa):
    num_iiwa_positions = plant.num_positions(iiwa)

    local_builder = DiagramBuilder()

    controller_plant = MultibodyPlant(time_step=TIME_STEP)
    controller_iiwa = AddIiwa(controller_plant, collision_model='with_box_collision')
    AddWsg(controller_plant, controller_iiwa, welded=True, sphere=True)
    controller_plant.Finalize()

    iiwa_state_port = local_builder.AddSystem(PassThrough(num_iiwa_positions *2))
    iiwa_state_port.set_name('iiwa_state_port')
    demux = local_builder.AddSystem(Demultiplexer(size=num_iiwa_positions*2,
                                                  output_ports_size=num_iiwa_positions))

    iiwa_controller = local_builder.AddSystem(
                InverseDynamicsController(controller_plant,
                                          kp=[100] * num_iiwa_positions,
                                          ki=[1] * num_iiwa_positions,
                                          kd=[20] * num_iiwa_positions,
                                          has_reference_acceleration=False))
    iiwa_controller.set_name('inner_iiwa_controller')

    desired_state_from_position = local_builder.AddSystem(
                StateInterpolatorWithDiscreteDerivative(
                    num_iiwa_positions,
                    TIME_STEP,
                    suppress_initial_transient=True))
    desired_state_from_position.set_name('iiwa_desired_state_from_position')

    local_builder.Connect(desired_state_from_position.get_output_port(),
                          iiwa_controller.get_input_port_desired_state())
    local_builder.Connect(iiwa_state_port.get_output_port(), iiwa_controller.get_input_port_estimated_state())
    local_builder.Connect(iiwa_state_port.get_output_port(), demux.get_input_port())

    local_builder.ExportInput(desired_state_from_position.get_input_port(), 'iiwa_position_desired')
    local_builder.ExportInput(iiwa_state_port.get_input_port(), 'iiwa_state')
    local_builder.ExportOutput(iiwa_controller.get_output_port_control(), 'iiwa_control')
    local_builder.ExportOutput(demux.get_output_port(0), 'iiwa_position_measured')

    diagram = local_builder.Build()
    diagram.set_name("iiwa_controller")
    return diagram


def make_gripper_frames(X_G, X_O, meshcat: typing.Optional[Meshcat] = None):

    p_GgraspO = [0.05, 0.07, 0.]
    p_Ggrasp2O = [0., 0.03, 0.03]

    R_GgraspO = RotationMatrix.MakeZRotation(np.pi/2.0)
    R_Ggrasp2O = RotationMatrix.MakeZRotation(-np.pi/2.0)
    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
    X_Ggrasp2O = RigidTransform(R_Ggrasp2O, p_Ggrasp2O)
    
    X_OGgrasp = X_GgraspO.inverse()

    # pregrasp is negative y in the gripper frame (see the figure!).
    X_GgraspGpregrasp = RigidTransform([0, -0.2, 0.0])
    X_Ggrasp2Gpregrasp = RigidTransform([0, -0.22, 0.03])

    X_Ggrasp3Gpregrasp = RigidTransform([0, -0.2, 0.03])
    X_Ggrasp4Gpregrasp = RigidTransform([0, -0.2, -0.07]) #works good [0, -0.2, 0.13]

    X_G['pick'] = X_O['initial'].multiply(X_OGgrasp)
    X_G['prepick'] = X_G['pick'].multiply(X_GgraspGpregrasp)
    X_G['postpick'] = X_G['pick'].multiply(X_Ggrasp3Gpregrasp)

    X_G['midway'] = X_G['pick'].multiply(X_Ggrasp4Gpregrasp)

    X_G['place'] = X_O['goal'].multiply(X_Ggrasp2O)
    X_G['preplace'] = X_G['place'].multiply(X_Ggrasp2Gpregrasp)
    X_G['postplace'] = X_G['preplace']
    X_G['final'] = X_G['initial']

    # I'll interpolate a halfway orientation by converting to axis angle and halving the angle.
    X_GpickGplace = X_G['pick'].inverse().multiply(X_G['place'])

    # Now let's set the timing
    times = {'initial': 0}
      
    X_GinitialGprepick = X_G['initial'].inverse().multiply(X_G['prepick'])
    times['prepick'] = times['initial'] + 10.0

    # Allow some time for the gripper to close.
    times['pick_start'] = times['prepick'] + 5
    X_G['pick_start'] = X_G['pick']
    
    times['pick_end'] = times['pick_start'] + 2.0
    X_G['pick_end'] = X_G['pick']
    times['postpick'] = times['pick_end'] + 2.0

    times['midway'] = times['postpick'] + 2.5
    times['preplace'] = times['midway'] + 2.5
      
    times['place_start'] = times['preplace'] + 2.0
    X_G['place_start'] = X_G['place']
      
    times['place_end'] = times['place_start'] + 4.0
    X_G['place_end'] = X_G['place']

    times['postplace'] = times['place_end'] + 4.0
    times['final'] = times['postplace'] + 10.0

    if meshcat:
        AddMeshcatTriad(meshcat, 'X_Ginitial', X_PT=X_G['initial'])

        AddMeshcatTriad(meshcat, 'X_Gprepick', X_PT=X_G['prepick'])
        AddMeshcatTriad(meshcat, 'X_Gpick', X_PT=X_G['pick'])
        AddMeshcatTriad(meshcat, 'X_Gpostpick', X_PT=X_G['postpick'])

        AddMeshcatTriad(meshcat, 'X_Gmidway', X_PT=X_G['midway'])

        AddMeshcatTriad(meshcat, 'X_Gpreplace', X_PT=X_G['preplace'])
        AddMeshcatTriad(meshcat, 'X_Gplace', X_PT=X_G['place'])
        AddMeshcatTriad(meshcat, 'X_Gpostlace', X_PT=X_G['postplace'])

    return X_G, times


def AddMeshcatTriad(
    meshcat, path, length=0.25, radius=0.01, opacity=1.0, X_PT=RigidTransform()
):
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


def create_wsg_position_desired_port(builder, plant, wsg):
    wsg_controller = builder.AddSystem(SchunkWsgPositionController())
    wsg_controller.set_name('wsg_controller')
    builder.Connect(plant.get_state_output_port(wsg),
                    wsg_controller.get_state_input_port())
    builder.Connect(wsg_controller.get_generalized_force_output_port(),
                    plant.get_actuation_input_port(wsg))

    return wsg_controller.get_desired_position_input_port()


def run_manipulation(method: str):
    meshcat = Meshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=TIME_STEP)
    iiwa = AddIiwa(plant, collision_model='with_box_collision')
    wsg = AddWsg(plant, iiwa, welded=False, sphere=False)

    shelves = get_resource(plant, 'shelves')
    X_WS = RigidTransform(RotationMatrix.Identity(), [0.7, 0., 0.4085])
    plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName('shelves_body', shelves),
            X_WS)

    brick_model = get_resource(plant, 'foam_brick')
    brick_body = plant.GetBodyByName('base_link', brick_model)
    X_WB = RigidTransform(RotationMatrix.Identity(), [0.6, 0., 0.4085 + 0.15])
    plant.SetDefaultFreeBodyPose(brick_body, X_WB)
    plant.Finalize()

    iiwa_controller = builder.AddSystem(create_iiwa_controller(plant, iiwa))
    iiwa_state_port = iiwa_controller.GetInputPort('iiwa_state')
    iiwa_control = iiwa_controller.GetOutputPort('iiwa_control')
    desired_iiwa_position_port = iiwa_controller.GetInputPort('iiwa_position_desired')
    measured_iiwa_position_port = iiwa_controller.GetOutputPort('iiwa_position_measured')
    
    builder.Connect(iiwa_control, plant.get_actuation_input_port(iiwa))
    builder.Connect(plant.get_state_output_port(iiwa), iiwa_state_port)
    desired_wsg_position_port = create_wsg_position_desired_port(builder, plant, wsg)
    #########
    # meshcat
    dst_translation = [0.6, 0., 0.2615 + 0.05]
    AddMeshcatSphere(meshcat, 'goal-meshcat', dst_translation)
    #########
    temp_plant_context = plant.CreateDefaultContext()
    X_G = {'initial': plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName('body'))}
    X_O = {'initial': plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName('base_link'))}

    X_O['goal'] = copy.deepcopy(X_O['initial'])
    X_O['goal'].set_translation(dst_translation)
    X_G, times = make_gripper_frames(X_G, X_O, meshcat)

    #########
    if 'inv-kin' == method:
        output_iiwa_position_port, output_wsg_position_port, integrator = \
            create_differential_controller(builder, plant,
                                           measured_iiwa_position_port,
                                           X_G, times)
    elif 'global' == method:
        integrator = None
        output_iiwa_position_port, output_wsg_position_port = create_open_loop_controller(builder, plant, iiwa, X_G, times)

    builder.Connect(output_iiwa_position_port, desired_iiwa_position_port)
    builder.Connect(output_wsg_position_port, desired_wsg_position_port)
    #########

    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    diagram.set_name(sys.argv[0])
    pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[0].write_png('diagram.png')


    simulator = Simulator(diagram)
    if not simulator:
        return

    plant.mutable_gravity_field().set_gravity_vector([0, 0, -1.])
    simulator.Initialize()
    if integrator is not None:
        integrator.set_integral_value(
            integrator.GetMyContextFromRoot(simulator.get_mutable_context()),
                plant.GetPositions(plant.GetMyContextFromRoot(simulator.get_mutable_context()),
                                   plant.GetModelInstanceByName('iiwa7')))

    web_url = meshcat.web_url()
    print(f'Meshcat is now available at {web_url}')
    os.system(f'xdg-open {web_url}')

    total_time = max(times.values())
    print(total_time)
    visualizer.StartRecording(False)
    simulator.AdvanceTo(total_time)
    visualizer.PublishRecording()
    time.sleep(30)


class ManipulationArgs(Tap):
    method: typing.Literal['inv-kin', 'global']  # Which controller to use

    def configure(self):
        self.add_argument('-m', '--method')


if __name__ == '__main__':
    run_manipulation(**ManipulationArgs().parse_args().as_dict())
