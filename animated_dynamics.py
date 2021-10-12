import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from feature_extraction import quaternion_to_euler_angle
from data_operations import load_data, body_movement_code

dataset = load_data()
for (body, rep), observation in dataset.groupby(level=(0, 1)):
    eulers = quaternion_to_euler_angle(observation[["qw", "qx", "qy", "qz"]])
    eulers = eulers.values
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    (line,) = ax.plot([], [], [], lw=2)
    ax.set_xlim3d([-3, 3])
    ax.set_ylim3d([-3, 3])
    ax.set_zlim3d([-3, 3])
    title = f'{body_movement_code[body]}, rep {rep}'    

    def update(num, line, data):
        line.set_data(data[:num, 0], data[:num, 1])
        line.set_3d_properties(data[:num, 2])
        return [line]


    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig,
        update,
        fargs=(line, eulers),
        interval=1,
        save_count=len(eulers),
    )

    plt.show()