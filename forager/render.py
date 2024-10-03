import argparse
import os
import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import display, HTML


def load_data(data_dir):
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
    data_files = sorted(data_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    data_dict = {}
    for data_file in data_files:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        if 'step' not in data:
            data['step'] = int(data_file.split('_')[-1].split('.')[0])
        else:
            data['step'] = int(data['step'])
        data_dict[data['step']] = data
    return data_dict


def convert_to_rgb(state, channels='wft'):
    rgb_state = np.ones((state.shape[0], state.shape[1], 3))
    rgb_state *= 255
    for idx, channel in enumerate(channels):
        if channel == 'w':
            rgb_state[state[:, :, idx] == 1] = [43,61,38]
        elif channel == 'f':
            rgb_state[state[:, :, idx] == 1] = [141,182,0]
        elif channel == 't':
            rgb_state[state[:, :, idx] == 1] = [190,0,50]
        elif channel == 'n':
            pass
        else:
            raise ValueError(f"Invalid channel {channel}")
    return rgb_state


def convert_coords(coords, size=500):
    # return (size - coords[1] - 1, coords[0])
    return (coords[0], size - coords[1] - 1)


def get_aperture(state, agent_coords, aperture_size=9):
    # pad the state with 0s
    pad_width = (aperture_size - 1) // 2
    state = np.pad(state, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)),
                   mode='constant', constant_values=0)
    aperture = state[agent_coords[0]:agent_coords[0] + aperture_size,
                     agent_coords[1]:agent_coords[1] + aperture_size]
    return convert_to_rgb(aperture)


def local_view(state, agent_coords, channels, aperture=16, agent_aperture=9,
               agent_size=10, step=0, start = 50000000):
    fig = plt.figure(figsize=(5, 5))
    fig_plot = fig.add_subplot(111)

    fig_plot.clear()

    rgb_state = convert_to_rgb(state, channels=channels)
    # pad the state with 0s
    rgb_state = np.pad(rgb_state, ((1, 1), (1, 1), (0, 0)),
                    mode='constant', constant_values=0)

    fig_plot.matshow(rgb_state.astype(np.uint8) / 255)

    agent_coords = convert_coords(agent_coords, size=state.shape[1])
    agent_coords = (agent_coords[0] + 1, agent_coords[1] + 1)

    # set axis limits to focus on the agent
    fig_plot.set_xlim(agent_coords[0] - aperture // 2 - 0.5,
                      agent_coords[0] + aperture // 2 + 0.5)
    fig_plot.set_ylim(agent_coords[1] - aperture // 2 - 0.5,
                      agent_coords[1] + aperture // 2 + 0.5)

    fig_plot.locator_params(nbins=1)
    fig_plot.invert_yaxis()

    # put a red dot, size 10, at the center of the aperture
    fig_plot.plot(agent_coords[0], agent_coords[1], 'o', ms=agent_size, color='#0067a5')
    # put a bounding box around the ceter with width agent_aperture
    fig_plot.add_patch(plt.Rectangle((agent_coords[0] - agent_aperture // 2 - 0.5,
                                      agent_coords[1] - agent_aperture // 2 - 0.5),
                                     agent_aperture, agent_aperture, ls='--',
                                     fill=False, edgecolor='#0067a5', lw=1.5))
    fig_plot.set_xlabel(f"Step: {step + start + 1}", fontdict={'fontsize': 12})

    # Remove ticks
    fig_plot.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    # Remove tick labels
    fig_plot.set_xticklabels([])
    fig_plot.set_yticklabels([])

    return fig

def full_view(state):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--aperture', type=int, default=10)
    parser.add_argument('--agent_aperture', type=int, default=3)
    parser.add_argument('--agent_size', type=int, default=20)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--channels', type=str, default=None)
    parser.add_argument('--local_view', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    agent_name = os.path.basename(args.data_dir)
    env_name = os.path.basename(os.path.dirname(args.data_dir))
    exp_dir = os.path.dirname(os.path.dirname(args.data_dir))

    save_path = f'{exp_dir}/{env_name}/{agent_name}/imgs'
    os.makedirs(save_path, exist_ok=True)

    data_dict = load_data(args.data_dir)
    # check if if all the keys are numbers and sorted
    keys = list(data_dict.keys())
    assert all(isinstance(key, int) for key in keys) and keys == sorted(keys), \
        f"Keys must be integers and sorted, got {keys}"

    # filter out the keys that are out of range
    if args.start is not None:
        keys = [key for key in keys if key >= args.start]
    if args.end is not None:
        keys = [key for key in keys if key <= args.end]

    # prep
    sample_data = data_dict[keys[-1]]
    if sample_data['states'].shape[-1] == 4:
        channels = 'wftw'
    elif sample_data['states'].shape[-1] == 3:
        channels = 'wft'
    elif sample_data['states'].shape[-1] == 2:
        channels = 'ft'
    elif sample_data['states'].shape[-1] == 1:
        channels = 'w'
    else:
        raise ValueError(f"Invalid number of channels {sample_data['states'].shape[-1]}")

    data = data_dict[keys[-1]]

    for i in tqdm.tqdm(range(args.frames), total=args.frames):
        state = data['states'][i]
        agent_coords = data['agent_coords'][i]

        if local_view:
            fig = local_view(state, agent_coords, channels, aperture=args.aperture, step=i,
                             agent_aperture=args.agent_aperture, agent_size=args.agent_size)
        else:
            fig = full_view(state)

        fig.savefig(f'{save_path}/{i}.pdf', format="pdf", bbox_inches="tight", dpi=300)

if __name__ == '__main__':
    main()
