import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

parser = argparse.ArgumentParser(description='Browse the game states stored in gameStates.npy')
parser.add_argument('-f', '--file', type=str, default='Data/gameStates.npy', required=False)
args = parser.parse_args()

states = np.load(args.file, mmap_mode='r')

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.15)
image = ax.imshow(states[0], interpolation='nearest')
ax.set_title('state 0/%d (red=fired, blue=boat, magenta=hit)' % (len(states) - 1))

slider = Slider(plt.axes([0.15, 0.05, 0.7, 0.04]), 'state', 0, len(states) - 1, valinit=0, valstep=1)

def Update(val):
    idx = int(slider.val)
    image.set_data(states[idx])
    ax.set_title('state %d/%d (red=fired, blue=boat, magenta=hit)' % (idx, len(states) - 1))
    fig.canvas.draw_idle()

slider.on_changed(Update)
plt.show()
