import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from feature_extraction import quaternion_to_euler_angle
from helpers import load_data, body_movement_code

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

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    writergif = animation.PillowWriter(fps=2500)
    anim.save(f'{title}.gif', writer=writergif)
    print(f'Finished {title}')