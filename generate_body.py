import pyrosim.pyrosim as ps


def create_elephant():
    ps.Start_URDF("elephant.urdf")

    torso_length = 2
    torso_width = 2
    torso_height = 1.5

    torso_pos = [1, 0, 3]
    ps.Send_Cube(name="Torso", pos=torso_pos, size=[torso_length, torso_width, torso_height])

    head_length = 1.5
    head_width = 1.5
    head_height = 1.5

    head_pos = [1, 0, -0.25]
    ps.Send_Cube(name="Head", pos=head_pos, size=[head_length, head_width, head_height])

    trunk_length = 0.5
    trunk_width = 0.5
    trunk_height = 2

    trunk_pos = [0.75, 0, -2.75]
    ps.Send_Cube(name="Trunk", pos=trunk_pos, size=[trunk_length, trunk_width, trunk_height])

    leg_length = 0.75
    leg_width = 0.75
    leg_height = 2

    leg_positions = [
        [0, 0.5, -1.5],     # Front left
        [0, -0.5, -1.5],    # Front right
        [1, 0.5, -1.5],     # Back left
        [1, -0.5, -1.5],    # Back right
    ]

    for i, pos in enumerate(leg_positions):
        foot_name = f"Foot_{i + 1}"
        ps.Send_Cube(name=foot_name, pos=pos, size=[leg_length, leg_width, leg_height])

        joint_name = f"Joint_{i + 1}"
        ps.Send_Joint(name=joint_name, parent="Torso", child=foot_name,
                      type="revolute", position=[pos[0], pos[1], torso_pos[2] + torso_height / 2])

        ps.Send_Sensor_Neuron(name=f"Touch_Sensor_{i + 1}", linkName=foot_name)

    ps.Send_Joint(name="Head_Joint", parent="Torso", child="Head",
                  type="revolute", position=[torso_pos[0], torso_pos[1], torso_pos[2] + torso_height / 2 + head_height / 2])

    ps.Send_Joint(name="Trunk_Joint", parent="Head", child="Trunk",
                  type="revolute", position=[head_pos[0], head_pos[1], head_pos[2] + head_height / 2 + trunk_height / 2])

    ps.End()


create_elephant()
