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
)

from grading import get_start_and_end_positions
from differential_controller import create_differential_controller
from open_loop_controller import create_open_loop_controller
from resource_loader import AddIiwa, AddWsg, get_resource, BRICK_GOAL_TRANSLATION
from visualization_tools import AddMeshcatSphere

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
    AddMeshcatSphere(meshcat, 'goal-meshcat', BRICK_GOAL_TRANSLATION)
    temp_plant_context = plant.CreateDefaultContext()
    X_G, X_O = get_start_and_end_positions(plant, temp_plant_context)
    #########

    if 'inv-kin' == method:
        output_iiwa_position_port, output_wsg_position_port, integrator, total_time = \
            create_differential_controller(builder, plant,
                                           measured_iiwa_position_port,
                                           X_G, X_O, meshcat)
    elif 'global' == method:
        integrator = None
        output_iiwa_position_port, output_wsg_position_port, total_time = create_open_loop_controller(builder, plant, iiwa, X_G, X_O)

    builder.Connect(output_iiwa_position_port, desired_iiwa_position_port)
    builder.Connect(output_wsg_position_port, desired_wsg_position_port)
    #########

    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    diagram.set_name(sys.argv[0])
    for depth in range(1, 3):
        pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=depth))[0].write_png(f'diagram-{depth}.png')


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
