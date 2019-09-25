from typing import Dict, Optional, Tuple

import glob
import os, sys
import tempfile

import numpy as np
import visdom
import torch
import torchvision
import colorlover as cl
from prettytable import PrettyTable
import PIL

import plotly.plotly as py
import plotly.figure_factory as ff
# from plotly.figure_factory._annotated_heatmap import _AnnotatedHeatmap
import plotly.graph_objs as go
from plotly import tools
import plotly.io as pio

import matplotlib
matplotlib.use('Agg')


from distutils.version import LooseVersion

_vis_min_version = '0.1.8.8'
assert LooseVersion(visdom.__version__) >= LooseVersion(_vis_min_version),\
    "Visdom version {} < supported {}".format(
        visdom.__version__, _vis_min_version)
# import moviepy.editor as mpy
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# matplotlib.rcParams.update({'font.size': 8})

def softmax(z):
    # assert len(z.shape) == 2
    # s = np.max(z, axis=1)[:, np.newaxis]
    e_x = np.exp(z - z.max())
    div = np.sum(e_x) #[:, np.newaxis]
    return e_x / div


def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]


def make_quiver_from_action_probs(action_probs, action_space_type,
    actions, title, bg_img=None):
    H, W = action_probs.shape[-2:]
    action_probs = action_probs * (action_probs > 0)
    _x, _y = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5)
    _u_zero, _v_zero = np.meshgrid(np.zeros(H), np.zeros(W))

    if action_space_type == 'pov':
        # POV action space (processed)
        # _pad = np.zeros(action_probs.shape)
        # action_probs = np.concatenate([_pad[:, 0:1], action_probs], 1)
        '''
        NOTE: Turn-left, Turn-right, as well as Down, Up are inverted
        here because the environment grid being viewed in visdom has
        been transposed.
        '''
        _right = action_probs[0]
        _down = -1 * action_probs[3]
        _left = -1 * action_probs[2]
        _up = action_probs[1]
        _turn_left = action_probs[5]
        _turn_right = action_probs[4]
        _pickup = action_probs[6]
        _drop = action_probs[7]
        _toggle = action_probs[8]
        _stay = action_probs[9]
        _stay = _stay + _drop + _toggle + _pickup

        # Uncomment to visualize maximum size of markers
        # _toggle[0, 0] = 1.0
        # _pickup[0, 1] = 1.0
        # _drop[0, 2] = 1.0
        # _stay[0, 3] = 1.0
        # _right[0, 4] = 1.0
        # _left[0, 5] = -1.0
        # _up[0, 6] = 1.0
        # _down[0, 7] = -1.0

    elif action_space_type == 'cardinal':
        # Cardinal action space
        '''
        NOTE: Down and Up are inverted here because the environment
        grid being viewed in visdom has been transposed, so the up action
        actually corresponds to moving down in the visualized grid.
        '''
        _right = action_probs[actions.right]
        _down = -1 * action_probs[actions.up]
        _left = -1 * action_probs[actions.left]
        _up = action_probs[actions.down]
        _stay = action_probs[actions.done]
        # _u = _right - _left
        # _v = _up - _down

        # Uncomment to visualize maximum size of markers
        # _stay[0, 1] = 1.0
        # _right[0, 0] = 1.0
        # _left[0, 0] = -1.0
        # _up[0, 0] = 1.0
        # _down[0, 0] = -1.0

    else:
        raise ValueError("Cannot recognize action space {}"\
            .format(action_space_type))

    common_kwargs = {'scale':0.5, 'arrow_scale': 0.5}
    quiver_1 = ff.create_quiver(
        _x, _y, _right, _v_zero, name='right', **common_kwargs)
    quiver_2 = ff.create_quiver(
        _x, _y, _left, _v_zero, name='left', **common_kwargs)
    quiver_3 = ff.create_quiver(
        _x, _y, _u_zero, _up, name='up', **common_kwargs)
    quiver_4 = ff.create_quiver(
        _x, _y, _u_zero, _down, name='down', **common_kwargs)
    quiver_1['data'][0]['marker'].update({'color':'blue'})
    quiver_2['data'][0]['marker'].update({'color':'orange'})
    quiver_3['data'][0]['marker'].update({'color':'green'})
    quiver_4['data'][0]['marker'].update({'color':'red'})

    def my_scatter(inp, title, symbol, fillcolor, linecolor):
        return go.Scatter(
            x=_x.reshape(-1),
            y=_y.reshape(-1),
            mode='markers',
            marker=dict(
                symbol=symbol,
                size=inp.reshape(-1),
                sizemode='area',
                opacity=0.4,
                sizeref= 2.0 / ((16**2) * (25./H)),
                line = dict(width = 1, color=linecolor),
                # color = 'rgba(204, 0, 204, .9)',
                color=fillcolor,
            ),
            name=title
        )
    quiver_5 = my_scatter(_stay, 'stay',
        symbol='octagon', fillcolor='red', linecolor='black')

    data = [
        *quiver_1['data'],
        *quiver_2['data'],
        *quiver_3['data'],
        *quiver_4['data'],
        quiver_5,
    ]

    if action_space_type == 'pov':
        # quiver_6 = my_scatter(_pickup, 'pickup',
        #     symbol='triangle-up', fillcolor='skyblue', linecolor='blue')
        # quiver_7 = my_scatter(_drop, 'drop',
        #     symbol='triangle-down', fillcolor='orange', linecolor='brown')
        # quiver_8 = my_scatter(_toggle, 'toggle',
        #     symbol='star', fillcolor='gold', linecolor='black')
        # data.extend([quiver_6, quiver_7, quiver_8])

        '''Left and right turns'''
        quiver_6 = my_scatter(_turn_left, 'turn_left',
            symbol='triangle-left', fillcolor='skyblue', linecolor='blue')
        quiver_7 = my_scatter(_turn_right, 'turn_right',
            symbol='triangle-right', fillcolor='orange', linecolor='brown')
        data.extend([quiver_6, quiver_7])
        pass

    if bg_img is None:
        fig = go.Figure()
    else:
        from PIL import Image
        # image_array = np.random.randint(0, 255, size=(100, 100)).astype('uint8')
        # image_tmp = Image.fromarray(image_array)
        bg_pil_img = Image.fromarray(np.rollaxis(bg_img, 0, 3))
        fig = go.Figure(
            layout=go.Layout(
                images=[go.layout.Image(
                    source= bg_pil_img,
                    xref= "x",
                    yref= "y",
                    x= 0,
                    y= H,
                    sizex= W,
                    sizey= H,
                    sizing= "stretch",
                    opacity= 0.1,
                    layer= "below",
                )],
                xaxis=dict(
                    showgrid=False,
                    showline=False,
                ),
                yaxis=dict(
                    showgrid=False,
                    showline=False,
                ),

            )
        )
    fig['layout']['title'] = title
    fig.add_traces(data=data)
    return fig


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


class VisdomLogger():
    def __init__(self,
                 env_name: str = 'main',
                 port: int = 8893,
                 server:str = "http://localhost",
                 log_file: Optional[str] = None,
                 fig_save_dir: Optional[str] = None,
                 win_prefix: str = "",
                 delete_existing_windows=True,
    ):
        '''
            Initialize a visdom server on $server:$port
        '''
        print("Initializing visdom env [%s]"%env_name)

        self.viz = visdom.Visdom(
            port = port,
            env = env_name,
            server = server,
            log_to_filename = log_file,
            username="FirstName",
            password="FirstName",
        )
        self.wins = {}
        self.win_prefix = win_prefix
        self.fig_save_dir = fig_save_dir
        self.delete_existing_windows = delete_existing_windows
        self.OVERWRITE_EXISTING_WINDOWS = True
        self.SAVE_RENDERED_FIGS = False
        if self.delete_existing_windows:
            self.viz.close()

        # if self.fig_save_dir is not None:
        #     if os.path.isfile("/tmp/.X8894-lock"):
        #         # Xvfb is up
        #         pass
        #     else:
        #         try:
        #             import subprocess
        #             self.xvfb_process = subprocess.Popen(
        #                 "Xvfb :8894 -screen 0 800x600x16", shell=True)
        #         except:
        #             print("Could not start xvfb process!")

        if "DISPLAY" not in os.environ:
            self.SAVE_RENDERED_FIGS = False
            print("Cannot save rendered figures if DISPLAY is not set!")

        self.toTensor = lambda x: x.cpu().detach().numpy()

    @staticmethod
    def clipValue(val, max=1e6, min=-1e6):
        if isinstance(val, np.ndarray):
            if np.isnan(val).any(): # NaN value check
                return np.zeros_like(val)
            else:
                return np.clip(val, min, max)
        else:
            if val != val:
                return 0
            else:
                return np.clip(val, min, max)

    def save_plotly_fig(self, iter_id, fig, sub_dir):
        if self.fig_save_dir:
            sub_dir = sub_dir[:].replace(" ", "_")
            root = os.path.join(self.fig_save_dir, sub_dir)
            os.makedirs(root, exist_ok=True)
            file_name = "{:09d}".format(iter_id)
            path = os.path.join(root, file_name + ".json")
            pio.write_json(fig, path)
            if self.SAVE_RENDERED_FIGS:
                path = os.path.join(root, file_name)
                pio.write_image(fig, path + ".png")
                pio.write_image(fig, path + ".svg")

    def line(
        self,
        x: float,
        y: float,
        key: str,
        line_name: str,
        xlabel: str = "Iterations",
        dash: str = "solid",
        ylabel: Optional[str] = None
    ):
        '''
            Add or update a plot on the visdom server self.viz
            Argumens:
                x : Scalar -> X-coordinate on plot
                y : Scalar -> Value at x
                key : Name of plot/graph
                line_name : Name of line within plot/graph
                xlabel : Label for x-axis (default: # Iterations)

            Plots and lines are created if they don't exist, otherwise
            they are updated.
        '''
        dash = np.array([dash])
        win_title = key
        key = self.win_prefix + key
        if key in self.wins.keys():
            self.viz.line(
                X = np.array([x]),
                Y = np.array([self.clipValue(y)]),
                win = self.wins[key],
                update = 'append',
                name = line_name,
                opts = dict(showlegend=True, dash=dash, webgl=False),
            )
        else:
            if self.viz.win_exists(key) and self.OVERWRITE_EXISTING_WINDOWS:
                print("Overwriting existing window in visdom env: {}"\
                    .format(key))
            if ylabel is None:
                ylabel = win_title
            self.wins[key] = self.viz.line(
                X = np.array([x]),
                Y = np.array([y]),
                win = key,
                name = line_name,
                opts = {
                    'xlabel': xlabel,
                    # 'ylabel': ylabel,
                    'title': win_title,
                    'dash': dash,
                    'showlegend': True,
                    'webgl': False,
                }
            )

    def bar(self, x, bin_names, key, title=None):
        win_title = key
        key = self.win_prefix + key
        opts = {
            'rownames': bin_names,
        }
        if title is not None:
            opts['title'] = title

        if key in self.wins.keys():
            self.viz.bar(
                X = x,
                win = self.wins[key],
                opts = opts
            )
        else:
            self.wins[key] = self.viz.bar(
                X = x,
                opts = opts
            )

    def boxplot(self, x, bin_names, key):
        win_title = key
        key = self.win_prefix + key

        if key in self.wins.keys():
            self.viz.boxplot(
                X = x,
                win = self.wins[key],
                opts = {
                    'legend': bin_names,
                }
            )
        else:
            self.wins[key] = self.viz.boxplot(
                X = x,
                opts = {
                    'legend': bin_names,
                }
            )

    def heatmap(self, x, key, root_power=4, normalize=False):
        assert root_power > 0
        if normalize:
            x_normalized = x / x.max()
            x_pow = np.power(x_normalized, 1.0/root_power)
            # x_new = np.concatenate([x_normalized, x_pow], 1)
            x_new = x_pow
        else:
            x_new = x
        win_title = key
        key = self.win_prefix + key

        if key in self.wins.keys():
            self.viz.heatmap(
                X = x_new,
                win = self.wins[key],
                opts = {
                    'colormap': 'Reds',
                    'xmin': 0,
                    # 'xmax': x_new.max(),
                    'layoutopts':{
                        'plotly': {'title': win_title}
                    }
                }
            )
        else:
            if self.viz.win_exists(key) and self.OVERWRITE_EXISTING_WINDOWS:
                print("Overwriting existing window in visdom env: {}"\
                    .format(key))
            self.wins[key] = self.viz.heatmap(
                X = x_new,
                opts = {
                    'colormap': 'Reds',
                    'xmin': 0,
                    # 'xmax': x_new.max(),
                    'layoutopts':{
                        'plotly': {'title': win_title}
                    }
                }
            )

    def plotly_scatter(
        self,
        xy_dict,
        key='scatter',
        normalize=False,
        opacity=0.9,
        marker_opacity=0.5,
        z_effect='size',
        opacity_scaling=None,
    ):
        assert z_effect in ['size', 'opacity']
        assert opacity_scaling in [None, 'log']

        win_title = key
        key = self.win_prefix + key

        # zmax = max(hmap.max(), 1e-8)
        # if normalize:
        #     zmax = 1.0
        #     hmap /= max(hmap.max(), 1e-8)

        fig = go.Figure()
        fig['layout']['title'] = win_title
        for name, xyz_dict in xy_dict.items():
            if 'z' in xyz_dict:
                if z_effect == 'size':
                    if normalize:
                        marker = dict(
                            opacity=marker_opacity,
                            size=15.0 * xyz_dict['z']/(xyz_dict['z'].max() + 1e-9),
                        )
                    else:
                        marker = dict(
                            opacity=marker_opacity,
                            size=xyz_dict['z'],
                        )
                elif z_effect == 'opacity':
                    if opacity_scaling == 'log':
                        log_z = np.log(xyz_dict['z'])
                        opacity_vals = (log_z - log_z.min()) \
                            / (log_z.max() - log_z.min() + 1e-9)
                    else:
                        opacity_vals = xyz_dict['z']/(xyz_dict['z'].max() + 1e-9)

                    marker = dict(
                        opacity=opacity_vals,
                    )
                else:
                    raise ValueError
            else:
                marker = dict(opacity=marker_opacity)

            trace = go.Scattergl(
                x=xyz_dict['x'],
                y=xyz_dict['y'],
                mode='markers',
                name=name,
                marker=marker,
                opacity=opacity,
            )
            fig.add_trace(trace)

        if key in self.wins.keys():
            self.viz.plotlyplot(
                figure = fig,
                win = self.wins[key],
            )
        else:
            if self.viz.win_exists(key) and self.OVERWRITE_EXISTING_WINDOWS:
                print("Overwriting existing window in visdom env: {}"\
                    .format(key))
            self.wins[key] = self.viz.plotlyplot(
                figure = fig,
                win = key,
            )

    def plotly_mc_viz(self, xy_dict, key, **kwargs):
        def _height(xs):
            return np.sin(3 * xs)*.45+.55

        pos_dict = {}
        neg_dict = {}

        for name, xyz_dict in xy_dict.items():
            pos_dict[name] = {}
            neg_dict[name] = {}

            v_values = xyz_dict['y']
            pos_inds = v_values >= 0

            y_values = _height(xyz_dict['x'])

            pos_dict[name] = {
                'x': xyz_dict['x'][pos_inds],
                'y': y_values[pos_inds],
                'z': v_values[pos_inds],
            }
            neg_dict[name] = {
                'x': xyz_dict['x'][~pos_inds],
                'y': y_values[~pos_inds],
                'z': -v_values[~pos_inds],
            }

        self.plotly_scatter(xy_dict=pos_dict, key=key+"_posv", **kwargs)
        self.plotly_scatter(xy_dict=neg_dict, key=key+"_negv", **kwargs)


    def plotly_heatmap(self, hmap, key, normalize=True, colorscale='Reds'):
        win_title = key
        key = self.win_prefix + key

        zmax = max(hmap.max(), 1e-8)
        if normalize:
            zmax = 1.0
            hmap /= max(hmap.max(), 1e-8)

        trace = go.Heatmap(
            z = hmap,
            x = np.arange(hmap.shape[0]),
            y = np.arange(hmap.shape[0]),
            colorscale=colorscale,
        )

        fig = tools.make_subplots(rows=1, cols=1,
            subplot_titles=[win_title], print_grid=False)
        fig.append_trace(trace, 1, 1)

        if key in self.wins.keys():
            self.viz.plotlyplot(
                figure = fig,
                win = self.wins[key],
            )
        else:
            if self.viz.win_exists(key) and self.OVERWRITE_EXISTING_WINDOWS:
                print("Overwriting existing window in visdom env: {}"\
                    .format(key))
            self.wins[key] = self.viz.plotlyplot(
                figure = fig,
                win = key,
            )

    def plotly_quiver_plot(self,
        action_probs,
        action_space_type,
        actions,
        key,
        bg_img=None,
        save_figures=False,
        iter_id=0,
    ):
        fig = make_quiver_from_action_probs(
            action_probs=action_probs,
            action_space_type=action_space_type,
            actions=actions,
            title=key,
            bg_img=bg_img)

        key = self.win_prefix + key

        if key in self.wins.keys():
            self.viz.plotlyplot(
                figure = fig,
                win = self.wins[key],
            )
        else:
            if self.viz.win_exists(key) and \
                self.OVERWRITE_EXISTING_WINDOWS:
                print("Overwriting existing window in visdom env: {}"\
                    .format(key))
            self.wins[key] = self.viz.plotlyplot(
                figure = fig,
                win = key,
            )

        fig['layout']['height'] = 500
        fig['layout']['width'] = 450 + 50

        if save_figures:
            self.save_plotly_fig(fig=fig, sub_dir=key, iter_id=iter_id)

    def plotly_grid(
        self,
        plot_type,
        hmap_batch: np.ndarray,
        ncols: int,
        key: str,
        subplot_titles=None,
        normalize=True,
        normalize_mode=None,
        action_space_type=None,
        actions=None,
        colorscale='Reds',
        bg_img=None,
        iter_id: int = 0,
        save_figures=False,
    ):
        '''hmap: (B, H, W) size numpy array'''
        assert ncols > 0
        assert plot_type in ['bar', 'heatmap', 'quiver', 'scatter']
        ANNOTATED_HEATMAP = False
        if normalize_mode is not None:
            assert normalize_mode in ['probability']

        b_size = hmap_batch.shape[0]
        hmap_batch = self.clipValue(hmap_batch)
        nrows = b_size // ncols
        if b_size % ncols != 0:
            nrows += 1
        padding = (nrows * ncols) - b_size

        if subplot_titles is None:
            subplot_titles = range(b_size)
            subplot_titles = [key + "_" + str(item) \
                for item in subplot_titles]
        subplot_titles += [None] * padding

        fig = tools.make_subplots(rows=nrows, cols=ncols,
            subplot_titles=subplot_titles, print_grid=False)
        fig['layout']['title'] = key

        traces = []
        annotations = []

        def ann_axes_change(anns, id, font_size=8):
            if anns is not None:
                for ann in anns:
                    if hasattr(ann, 'xref'):
                        ann['xref'] = 'x{}'.format(id)
                    if hasattr(ann, 'yref'):
                        ann['yref'] = 'y{}'.format(id)
                    ann['font']['size'] = font_size

        # zmax = max(hmap_batch.max(), 1e-8)
        if plot_type == 'quiver':
            pass
        else:
            for b_idx in range(b_size):
                if normalize:
                    if normalize_mode == 'probability':
                        hmap_batch[b_idx] = softmax(hmap_batch[b_idx])
                    else:
                        hmap_batch[b_idx] /= max(hmap_batch[b_idx].max(), 1e-8)
            zmax = max(hmap_batch.max(), 1e-8)
            zmin = min(0, hmap_batch.min())

        for b_idx in range(b_size):
            row_id = 1 + (b_idx // ncols)
            col_id = 1 + (b_idx % ncols)

            if plot_type == 'heatmap':
                _hmap = go.Heatmap(
                    z=hmap_batch[b_idx],
                    colorscale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                )
                traces.append(_hmap)
                if ANNOTATED_HEATMAP:
                    _anns = _AnnotatedHeatmap(
                        z=np.round(hmap_batch[b_idx], decimals=2),
                        x=None, y=None,
                        annotation_text=None,
                        font_colors=None,
                        colorscale=colorscale,
                        reversescale=False
                    ).make_annotations()
                    # _tmp = go.Heatmap(
                    #     z=hmap_batch[b_idx],
                    #     colorscale='Reds',
                    #     zmin=zmin,
                    #     zmax=zmax,
                    # )
                    # _hmap = ff.create_annotated_heatmap(
                    #     z=hmap_batch[b_idx],
                    #     annotation_text=np.round(hmap_batch[b_idx], decimals=2),
                    #     colorscale='Reds',
                    #     zmin=zmin,
                    #     zmax=zmax,
                    # )
                    # _hmap['data'][0]['showscale'] = True
                    # traces.append(_hmap['data'][0])
                    # _anns = _hmap['layout']['annotations']
                    ann_axes_change(_anns, b_idx + 1)
                    annotations.extend(_anns)
                    # fig['layout']['xaxis{}'.format(b_idx + 1)].update(
                    #     _hmap['layout']['xaxis'])
                    # fig['layout']['yaxis{}'.format(b_idx + 1)].update(
                    #     _hmap['layout']['yaxis'])
                fig.append_trace(traces[b_idx], row_id, col_id)

            elif plot_type == 'bar':
                raise NotImplementedError
                traces.append(go.Bar(
                    x = np.arange(hmap_batch.shape[2]),
                    y = hmap_batch[b_idx][0],
                ))
                fig.append_trace(traces[b_idx], row_id, col_id)

            elif plot_type == 'quiver':
                quiver_fig = make_quiver_from_action_probs(
                    action_probs=hmap_batch[b_idx],
                    action_space_type=action_space_type,
                    actions=actions,
                    title=key,
                    bg_img=bg_img)
                # traces.append(quiver_fig['data'])
                for _trace in quiver_fig['data']:
                    fig.append_trace(_trace, row_id, col_id)

            elif plot_type == 'scatter':
                raise NotImplementedError
                # scatter_fig = go.Scatter(
                #     x=,
                #     y=,
                #     mode='markers',
                #     name=b_idx,
                # )
                traces.append(scatter_fig)
                fig.append_trace(traces[b_idx], row_id, col_id)

        if ANNOTATED_HEATMAP:
            fig['layout']['annotations'] = annotations

        key = self.win_prefix + key

        if key in self.wins.keys():
            self.viz.plotlyplot(
                figure = fig,
                win = self.wins[key],
            )
        else:
            if self.viz.win_exists(key) and \
                self.OVERWRITE_EXISTING_WINDOWS:
                print("Overwriting existing window in visdom env: {}"\
                    .format(key))
            self.wins[key] = self.viz.plotlyplot(
                figure = fig,
                win = key,
            )

        fig['layout']['height'] = 500 * nrows
        fig['layout']['width'] = (450 * ncols) + 50

        if save_figures:
            self.save_plotly_fig(fig=fig, sub_dir=key, iter_id=iter_id)


    def kl_table_and_hmap(
        self,
        key: str,
        agent_pos: np.ndarray,
        kl_values: np.ndarray,
        lld_values: np.ndarray,
        masks: np.ndarray,
        options: Optional[np.ndarray] = None,
    ):
        kl_values = self.clipValue(kl_values)
        lld_values = self.clipValue(lld_values)
        kl_values = kl_values * masks
        lld_values = lld_values * masks
        agent_pos = (agent_pos * masks) - (np.ones_like(agent_pos) * (1 - masks))

        kl_values = np.squeeze(kl_values)
        lld_values = np.squeeze(lld_values)
        masks = np.squeeze(masks)

        BATCH_LIMIT = 12
        kl_values = kl_values[:, :BATCH_LIMIT]
        lld_values = lld_values[:, :BATCH_LIMIT]
        agent_pos = agent_pos[:, :BATCH_LIMIT]
        masks = masks[:, :BATCH_LIMIT]
        batch_size = agent_pos.shape[1]

        CMAX = 500
        CMIN = 50
        color_scale = cl.scales['9']['seq']['Reds']
        color_scale_fine = cl.interp(color_scale, CMAX)

        kl_cids = ((CMAX - (CMIN + 1)) * (kl_values \
            / (1e-8 + 1.0 * kl_values.max(0))))
        kl_cids = ((CMIN + kl_cids) * masks).astype('int')
        # ll_cids = ((CMAX - (CMIN + 1)) * ((lld_values - lld_values.min(0)) \
        #     / (1.0 * (lld_values.max(0) - lld_values.min(0)))))
        # ll_cids = ((CMIN + ll_cids) * masks).astype('int')
        kl_colors = np.array(color_scale_fine)[kl_cids]
        kl_colors = [kl_colors[:, t] for t in range(batch_size)]

        pos_str = agent_pos.astype('int').astype('str')
        kl_str = np.around(kl_values, 2).astype('str')
        lld_str = np.around(lld_values, 2).astype('str')
        tmp = np.core.defchararray.add(pos_str[:, :, 0], ',')
        tmp = np.core.defchararray.add(tmp, pos_str[:, :, 1])
        tmp = np.core.defchararray.add(tmp, ' \ ')
        tmp = np.core.defchararray.add(tmp, kl_str)
        tmp = np.core.defchararray.add(tmp, ',')
        tmp = np.core.defchararray.add(tmp, lld_str)
        display_str = tmp
        display_str = [display_str[:, t] for t in range(batch_size)]

        if options is None:
            column_headers = ['<b>A{}</b>'.format(idx) \
                for idx in range(batch_size)][:batch_size]
        else:
            column_headers = ['<b>A{}-O{}</b>'.format(idx, opt) \
                for idx, opt in enumerate(options)][:batch_size]


        trace = go.Table(
            header = dict(
                # values = ['<b>Column A</b>', '<b>Column B</b>', '<b>Column C</b>'],
                values = column_headers,
                # line = dict(color = 'white'),
                # fill = dict(color = 'white'),
                align = 'center',
                font = dict(color = 'black', size = 12)
            ),
            cells = dict(
                values = display_str,
                line = dict(color = kl_colors),
                fill = dict(color = kl_colors),
                align = 'center',
                font = dict(color = 'white', size = 11),
            )
        )

        fig = go.Figure(data=[trace])

        # key = "kl_table_and_hmap"
        fig['layout']['title'] = key
        if key in self.wins.keys():
            self.viz.plotlyplot(
                figure = fig,
                win = self.wins[key],
            )
        else:
            if self.viz.win_exists(key) and self.OVERWRITE_EXISTING_WINDOWS:
                print("Overwriting existing window in visdom env: {}"\
                    .format(key))
            self.wins[key] = self.viz.plotlyplot(
                figure = fig,
                win = key,
            )

    def action_table(
        self,
        key: str,
        actions: np.ndarray,
        action_class,
        agent_pos: np.ndarray,
        reward: np.ndarray,
        masks: np.ndarray,
        bonus_tensor = None,
    ):
        # agent_pos = (agent_pos * masks) - (np.ones_like(agent_pos) * (1 - masks))
        bonus_t = bonus_tensor
        if bonus_tensor is None:
            bonus_t = masks * 0

        actions = np.squeeze(actions)
        reward = np.squeeze(reward)
        masks = np.squeeze(masks)
        bonus_t = np.squeeze(bonus_t)

        BATCH_LIMIT = 12
        actions = actions[:, :BATCH_LIMIT]
        reward = reward[:, :BATCH_LIMIT]
        agent_pos = agent_pos[:, :BATCH_LIMIT]
        agent_pos = agent_pos[:-1]
        masks = masks[:, :BATCH_LIMIT]
        masks = masks[:-1]
        bonus_t = bonus_t[:, :BATCH_LIMIT]
        bonus_t = bonus_t[:-1]
        batch_size = actions.shape[1]
        if bonus_tensor is not None:
            bonus_tensor

        CMAX = 500
        CMIN = 50
        color_sizes = reward
        if bonus_tensor is not None:
            color_sizes += bonus_t
        R_MAX = max(color_sizes.max(), 1e-8)
        color_scale = cl.scales['9']['seq']['Reds']
        color_scale_fine = cl.interp(color_scale, CMAX)

        r_cids = ((CMAX - (CMIN + 1)) * (color_sizes \
            / (1e-8 + 1.0 * R_MAX)))
        r_cids = ((CMIN + r_cids) * masks).astype('int')
        # ll_cids = ((CMAX - (CMIN + 1)) * ((lld_values - lld_values.min(0)) \
        #     / (1.0 * (lld_values.max(0) - lld_values.min(0)))))
        # ll_cids = ((CMIN + ll_cids) * masks).astype('int')
        r_cids = np.clip(r_cids, 0, len(color_scale_fine))
        r_colors = np.array(color_scale_fine)[r_cids]
        r_colors = [r_colors[:, t] for t in range(batch_size)]

        assert len(actions.shape) == 2
        action_dict = {action.value:action.name[:2] for action in action_class}

        to_action_str = np.vectorize(lambda x: action_dict[x])
        actions_str = to_action_str(actions)
        masks_str = masks.astype('int').astype('str')
        pos_str = agent_pos.astype('int').astype('str')
        reward_str = np.round(reward, 2).astype('str')
        bonus_t_str = np.round(bonus_t, 4).astype('str')

        tmp = np.core.defchararray.add(pos_str[:, :, 0], ',')
        tmp = np.core.defchararray.add(tmp, pos_str[:, :, 1])
        tmp = np.core.defchararray.add(tmp, '_')
        tmp = np.core.defchararray.add(tmp, actions_str)
        tmp = np.core.defchararray.add(tmp, '_')
        tmp = np.core.defchararray.add(tmp, masks_str)
        tmp = np.core.defchararray.add(tmp, '_')
        tmp = np.core.defchararray.add(tmp, reward_str)
        if bonus_tensor is not None:
            tmp = np.core.defchararray.add(tmp, '_')
            tmp = np.core.defchararray.add(tmp, bonus_t_str)

        display_str = tmp
        display_str = [display_str[:, t] for t in range(batch_size)]

        column_headers = ['<b>A{}</b>'.format(idx) \
            for idx in range(batch_size)][:batch_size]

        trace = go.Table(
            header = dict(
                values = column_headers,
                align = 'center',
                font = dict(color = 'black', size = 16)
            ),
            cells = dict(
                values = display_str,
                align = 'center',
                line = dict(color = r_colors),
                fill = dict(color = r_colors),
                font = dict(color = 'black', size = 11),
            )
        )

        fig = go.Figure(data=[trace])

        fig['layout']['title'] = key
        if key in self.wins.keys():
            self.viz.plotlyplot(
                figure = fig,
                win = self.wins[key],
            )
        else:
            if self.viz.win_exists(key) and self.OVERWRITE_EXISTING_WINDOWS:
                print("Overwriting existing window in visdom env: {}"\
                    .format(key))
            self.wins[key] = self.viz.plotlyplot(
                figure = fig,
                win = key,
            )

    def image(self, img_array: np.ndarray, key: str):
        assert len(img_array.shape) == 3
        assert img_array.shape[0] == 3

        caption = key
        key = self.win_prefix + key
        if key in self.wins.keys():
            self.viz.image(
                img = img_array,
                win = self.wins[key],
                opts=dict(caption=caption),
            )
        else:
            if self.viz.win_exists(key) and \
                self.OVERWRITE_EXISTING_WINDOWS:
                print("Overwriting existing window in visdom env: {}"\
                    .format(key))
            self.wins[key] = self.viz.image(
                img = img_array,
                win = key,
                opts=dict(caption=caption),
            )

    # def video(self, tensor, key, fps=2):
    #     win_title = key
    #     key = self.win_prefix + key
    #
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         for i in range(tensor.shape[0]):
    #             path = os.path.join(temp_dir, "%04d.jpg"%i)
    #             torchvision.utils.save_image(tensor[i], path)
    #
    #         file_list = glob.glob(os.path.join(temp_dir, "*.jpg"))
    #         file_list.sort()
    #
    #         videofile = os.path.join(temp_dir, "video.mp4")
    #         clip = mpy.ImageSequenceClip(file_list, fps=fps)
    #         clip.write_videofile(videofile, fps=fps)
    #         print("Saved video to file:", videofile)
    #
    #         if key in self.wins.keys():
    #             self.viz.video(
    #                 videofile = videofile,
    #                 win = self.wins[key],
    #                 # opts={'fps': fps},
    #             )
    #         else:
    #             self.wins[key] = self.viz.video(
    #                 videofile = videofile,
    #                 # opts={'fps': fps},
    #             )

    def text(self, text, key):
        win_title = key
        key = self.win_prefix + key

        if type(text) == dict:
            table = PrettyTable(['key', 'value'])
            for key in sorted(text.keys()):
                val = text[key]
                table.add_row([key, val])
            text = table.get_html_string()

        if self.viz.win_exists(key) and self.OVERWRITE_EXISTING_WINDOWS:
            print("Overwriting existing window in visdom env: {}"\
                .format(key))
        if key in self.wins.keys():
            self.viz.text(text, win=self.wins[key])
        else:
            self.wins[key] = self.viz.text(text)


    # def visdom_plot(key,
    #                 folder,
    #                 game,
    #                 name,
    #                 num_steps,
    #                 bin_size=100,
    #                 smooth=1):
    #     tx, ty = load_data(folder, smooth, bin_size)
    #
    #     if tx is None or ty is None:
    #         return None
    #
    #     fig = plt.figure()
    #     plt.plot(tx, ty, label="{}".format(name))
    #
    #     tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    #     ticks = tick_fractions * num_steps
    #     tick_names = ["{:.0e}".format(tick) for tick in ticks]
    #     plt.xticks(ticks, tick_names)
    #     plt.xlim(0, num_steps * 1.01)
    #
    #     plt.xlabel('Number of Timesteps')
    #     plt.ylabel('Rewards')
    #
    #     plt.title(game)
    #     plt.legend(loc=4)
    #     plt.show()
    #     plt.draw()
    #
    #     image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    #     plt.close(fig)
    #
    #     # Show it in visdom
    #     image = np.transpose(image, (2, 0, 1))
    #
    #     win_title = key
    #     key = self.win_prefix + key
    #     if key in self.wins.keys():
    #         self.viz.image(image, win=self.wins[key])
    #     else:
    #         self.wins[key] = self.viz.image(image, win=win)
