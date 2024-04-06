#!/usr/bin/env python3

import os
import sys
import time

from typing import Literal

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
    RotationMatrix,

    Adder,
    Demultiplexer,
    PassThrough,
    InverseDynamicsController,
)

from differential_controller import create_differential_controller


TIME_STEP=0.007  #faster
IIWA_DEFAULT_POSITION = [-1.57, 0.1, 0, -1.2, 0, 1.6, 0]

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


def get_resource_path(resource_name: str) -> str:
    resource_file = os.path.join('resources', f'{resource_name}.sdf')
    full_path = os.path.join(os.path.split(os.path.abspath(os.path.dirname(sys.argv[0])))[0], resource_file)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise Exception(f'a resource {resource_name} is absent')
    return full_path


def get_resource(plant: MultibodyPlant, resource_name: str):
    resource_path = get_resource_path(resource_name)
    return Parser(plant=plant).AddModelFromFile(resource_path)


def create_iiwa_position_measured_port(builder, plant, iiwa):
    num_iiwa_positions = plant.num_positions(iiwa)
    iiwa_output_state = plant.get_state_output_port(iiwa)
    demux = builder.AddSystem(Demultiplexer(size=num_iiwa_positions*2,
                                            output_ports_size=num_iiwa_positions))
    builder.Connect(plant.get_state_output_port(iiwa),
                    demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), "iiwa_position_measured")

    controller_plant = MultibodyPlant(time_step=TIME_STEP)
    controller_iiwa = AddIiwa(controller_plant, collision_model="with_box_collision")
    AddWsg(controller_plant, controller_iiwa, welded=True, sphere=True)
    controller_plant.Finalize()

    iiwa_controller = builder.AddSystem(
                InverseDynamicsController(controller_plant,
                                          kp=[100] * num_iiwa_positions,
                                          ki=[1] * num_iiwa_positions,
                                          kd=[20] * num_iiwa_positions,
                                          has_reference_acceleration=False))
    iiwa_controller.set_name("iiwa_controller")
    builder.Connect(plant.get_state_output_port(iiwa),
                    iiwa_controller.get_input_port_estimated_state())

    adder = builder.AddSystem(Adder(2, num_iiwa_positions))
    builder.Connect(iiwa_controller.get_output_port_control(),
                    adder.get_input_port(0))
    torque_passthrough = builder.AddSystem(PassThrough([0] * num_iiwa_positions))
    builder.Connect(torque_passthrough.get_output_port(),
                    adder.get_input_port(1))
    builder.ExportInput(torque_passthrough.get_input_port(), "iiwa_feedforward_torque")
    builder.Connect(adder.get_output_port(),
                    plant.get_actuation_input_port(iiwa))

    return demux.get_output_port(0), iiwa_controller, iiwa_output_state


def run_manipulation(method: str):
    meshcat = Meshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=TIME_STEP)
    iiwa = AddIiwa(plant, collision_model="with_box_collision")
    #wsg = AddWsg(plant, iiwa, welded=False, sphere=False)

    shelves = get_resource(plant, 'shelves')
    X_WS = RigidTransform(RotationMatrix.Identity(), [0.5, 0., 0.4085])
    plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName('shelves_body', shelves),
            X_WS)
    brick_model = get_resource(plant, 'foam_brick')
    brick_body = plant.GetBodyByName("base_link", brick_model)
    X_WB = RigidTransform(RotationMatrix.Identity(), [0.4, 0., 0.4085 + 0.15])
    plant.SetDefaultFreeBodyPose(brick_body, X_WB)
    plant.Finalize()

    measured_iiwa_position_port, iiwa_pid_controller, measured_iiwa_state_port = \
        create_iiwa_position_measured_port(
            builder, plant, iiwa)

    #########
    temp_plant_context = plant.CreateDefaultContext()
    X_G = {"initial": plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName("body"))}
    X_O = {"initial": plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName("nut"))}
    X_OinitialOgoal = RigidTransform(RotationMatrix.MakeZRotation(-np.pi / 6))
    X_O['goal'] = X_O['initial'].multiply(X_OinitialOgoal)
    X_G, times = diff2_c.make_gripper_frames(X_G, X_O)

    #########
    output_iiwa_position_port, output_wsg_position_port, integrator = \
        create_differential_controller(builder, plant,
                                       measured_iiwa_position_port,
                                       X_G, times)
    #########

    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    diagram.set_name(sys.argv[0])
    pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[0].write_png('diagram.png')

    simulator = Simulator(diagram)
    if not simulator:
        return

    simulator.Initialize()
    if integrator is not None:
        integrator.set_integral_value(
            integrator.GetMyContextFromRoot(simulator.get_mutable_context()),
                plant.GetPositions(plant.GetMyContextFromRoot(simulator.get_mutable_context()),
                                   plant.GetModelInstanceByName("iiwa7")))

    web_url = meshcat.web_url()
    print(f'Meshcat is now available at {web_url}')
    os.system(f'xdg-open {web_url}')

    total_time = 10
    print(total_time)
    visualizer.StartRecording(False)
    simulator.AdvanceTo(total_time)
    visualizer.PublishRecording()
    time.sleep(30)


class ManipulationArgs(Tap):
    method: Literal['inv-kin', 'global']  # Which controller to use

    def configure(self):
        self.add_argument('-m', '--method')


if __name__ == '__main__':
    run_manipulation(**ManipulationArgs().parse_args().as_dict())
