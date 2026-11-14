from tac.io.mjcf import load_mjcf

robot = load_mjcf("examples/gnoci_v2/robot.xml")
print(robot)