import os
# import tqdm
import socket
import pickle as pkl
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

# import pdb

from bmtk.simulator.filternet.lgnmodel.fitfuns import makeBasis_StimKernel
from bmtk.simulator.filternet.lgnmodel.spatialfilter import GaussianSpatialFilter
from bmtk.simulator.filternet.lgnmodel.temporalfilter import TemporalFilterCosineBump
from bmtk.simulator.filternet.lgnmodel.util_fns import get_data_metrics_for_each_subclass, \
    get_tcross_from_temporal_kernel


def create_temporal_filter(inp_dict):
    opt_wts = inp_dict['opt_wts']
    opt_kpeaks = inp_dict['opt_kpeaks']
    opt_delays = inp_dict['opt_delays']
    temporal_filter = TemporalFilterCosineBump(opt_wts, opt_kpeaks, opt_delays)

    return temporal_filter


def create_one_unit_of_two_subunit_filter(prs, ttp_exp):
    filt = create_temporal_filter(prs)
    tcross_ind = get_tcross_from_temporal_kernel(
        filt.get_kernel(threshold=-1.0).kernel)
    filt_sum = filt.get_kernel(threshold=-1.0).kernel[:tcross_ind].sum()

    # Calculate delay offset needed to match response latency with data and rebuild temporal filter
    del_offset = ttp_exp - tcross_ind
    if del_offset >= 0:
        delays = prs['opt_delays']
        delays[0] = delays[0] + del_offset
        delays[1] = delays[1] + del_offset
        prs['opt_delays'] = delays
        filt_new = create_temporal_filter(prs)
    else:
        print('del_offset < 0')

    return filt_new, filt_sum


def temporal_filter(all_spatial_responses, temporal_kernels):
    tr_spatial_responses = tf.pad(
        all_spatial_responses[None, :, None, :],
        ((0, 0), (temporal_kernels.shape[-1] - 1, 0), (0, 0), (0, 0)))

    tr_temporal_kernels = tf.transpose(temporal_kernels)[:, None, :, None]
    filtered_output = tf.nn.depthwise_conv2d(
        tr_spatial_responses, tr_temporal_kernels, strides=[1, 1, 1, 1], padding='VALID')[0, :, 0]
    return filtered_output


def transfer_function(_a):
    _h = tf.cast(_a >= 0, tf.float32)
    return _h * _a


def select_spatial(x, y, convolved_movie):
    i1 = np.stack((np.floor(y), np.floor(x)), -1).astype(np.int32)
    i2 = np.stack((np.ceil(y), np.floor(x)), -1).astype(np.int32)
    i3 = np.stack((np.floor(y), np.ceil(x)), -1).astype(np.int32)
    i4 = np.stack((np.ceil(y), np.ceil(x)), -1).astype(np.int32)
    # indices = np.stack((120 - y[sel], 240 - x[sel] - 1), -1)
    transposed_convolved_movie = tf.transpose(convolved_movie, (1, 2, 0))
    sr1 = tf.gather_nd(transposed_convolved_movie, i1)
    sr2 = tf.gather_nd(transposed_convolved_movie, i2)
    sr3 = tf.gather_nd(transposed_convolved_movie, i3)
    sr4 = tf.gather_nd(transposed_convolved_movie, i4)
    ss = tf.stack((sr1, sr2, sr3, sr4), 0)
    y_factor = (y - np.floor(y))
    x_factor = (x - np.floor(x))
    weights = np.array([
        (1 - x_factor) * (1 - y_factor),
        (1 - x_factor) * y_factor,
        x_factor * (1 - y_factor),
        x_factor * y_factor
    ])
    spatial_responses = tf.reduce_sum(ss * weights[..., None], 0)
    spatial_responses = tf.transpose(spatial_responses)
    return spatial_responses

def create_lgn_units_info(csv_path='data/lgn_node_types.csv', h5_path='data/lgn_nodes.h5', filename='lgn_full_col_cells.csv'):
    filename = os.path.join('data', filename)
    # Load both the h5 file and the csv file
    csv_file = pd.read_csv(csv_path, sep=' ')
    features = ['id', 'model_id', 'x', 'y', 'ei', 'location', 'spatial_size', 'kpeaks_dom_0', 'kpeaks_dom_1', 'weight_dom_0', 'weight_dom_1', 'delay_dom_0', 'delay_dom_1', 'kpeaks_non_dom_0', 'kpeaks_non_dom_1', 'weight_non_dom_0', 'weight_non_dom_1', 'delay_non_dom_0', 'delay_non_dom_1', 'tuning_angle', 'sf_sep']
    df = pd.DataFrame(columns=features)

    with h5py.File(h5_path, 'r') as h5_file:
        node_id = h5_file['nodes']['lgn']['node_id'][:]
        node_type_id = h5_file['nodes']['lgn']['node_type_id'][:]
        for feature in h5_file['nodes']['lgn']['0'].keys():
            df[feature] = h5_file['nodes']['lgn']['0'][feature][:]

    node_info = {}
    for index, row in csv_file.iterrows():
        node_info[row['node_type_id']] = {'model_id': row['pop_name'], 'location': row['location'], 'ei': row['ei']}

    df['id'] = node_id
    df['model_id'] = [node_info[node_type_id[i]]['model_id'] for i in range(len(node_type_id))]
    df['location'] = [node_info[node_type_id[i]]['location'] for i in range(len(node_type_id))]
    df['ei'] = [node_info[node_type_id[i]]['ei'] for i in range(len(node_type_id))]

    df.to_csv(filename, index=False, sep=' ', na_rep='NaN')
    return df


class LGN(object):
    def __init__(self, row_size=80, col_size=120, lgn_data_path=None):
        if lgn_data_path is None:
            lgn_data_path_name = f'lgn_full_col_cells_{col_size}x{row_size}.csv'
            hostname = socket.gethostname()
            # print current path
            print(os.getcwd())
            if hostname.count('nvcluster') > 0:
                path = f'/home/ifgovh/tf_billeh_column/{lgn_data_path_name}'
            elif hostname.count('juwels') > 0:
                path = f'/p/project/structuretofunction/guozhang/glif_criticality/GLIF_network/{lgn_data_path_name}'
            elif hostname.count('pCluster') > 0:
                path = f'/home/guozhang/allen/LGN/LGN/{lgn_data_path_name}'
            elif hostname.count('nid') > 0:
                path = f'/users/bp000436/glif_criticality/GLIF_network/{lgn_data_path_name}'
            elif hostname.count('shinya') > 0:
                path = f'/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/tensorflow/{lgn_data_path_name}'
            else:
                path = f'/home/jgalvan/Desktop/Neurocoding/Visual_stimulus/Utils/data/{lgn_data_path_name}'
            d = pd.read_csv(path, delimiter=' ')
        else:
            d = create_lgn_units_info()
        
        # print(d['spatial_size'])
        spatial_sizes = d['spatial_size'].to_numpy()
        self.spatial_sizes = spatial_sizes
        model_id = d['model_id'].to_numpy()
        self.model_id = model_id
        amplitude = np.array(
            [1. if a.count('ON') > 0 else -1. for a in model_id])
        non_dom_amplitude = np.zeros_like(amplitude)
        is_composite = np.array([a.count('ON') > 0 and a.count(
            'OFF') > 0 for a in model_id]).astype(np.float32)
        self.is_composite = is_composite
        x = d['x'].to_numpy()
        y = d['y'].to_numpy()
        non_dominant_x = np.zeros_like(x)
        non_dominant_y = np.zeros_like(y)
        tuning_angle = d['tuning_angle'].to_numpy()
        subfield_separation = d['sf_sep'].to_numpy()  # for composite cells

        root_path = os.path.split(__file__)[0]
        s_path = os.path.join(root_path, f'spontaneous_firing_rates_{col_size}x{row_size}.pkl')
        if not os.path.exists(s_path):
            cell_type = [a[:a.find('_')] for a in model_id]
            tf_str = [a[a.find('_') + 1:] for a in model_id]
            spontaneous_firing_rates = []
            print('Computing spontaneous firing rates')
            # for a, b in tqdm.tqdm(zip(cell_type, tf_str), total=len(model_id)):
            for a, b in zip(cell_type, tf_str):
                if a.count('ON') > 0 and a.count('OFF') > 0:
                    spontaneous_firing_rates.append(-1)
                else:
                    spontaneous_firing_rate = get_data_metrics_for_each_subclass(a)[
                        b]['spont_exp']
                    spontaneous_firing_rates.append(spontaneous_firing_rate[0])
            spontaneous_firing_rates = np.array(spontaneous_firing_rates)
            with open(s_path, 'wb') as f:
                pkl.dump(spontaneous_firing_rates, f)
                print('Caching spontaneous firing rates')
        else:
            with open(s_path, 'rb') as f:
                spontaneous_firing_rates = pkl.load(f)

        temporal_peaks_dom = np.stack(
            (d['kpeaks_dom_0'].to_numpy(), d['kpeaks_dom_1'].to_numpy()), -1)
        temporal_weights = np.stack(
            (d['weight_dom_0'].to_numpy(), d['weight_dom_1'].to_numpy()), -1)
        temporal_delays = np.stack(
            (d['delay_dom_0'].to_numpy(), d['delay_dom_1'].to_numpy()), -1)

        temporal_peaks_non_dom = np.stack(
            (d['kpeaks_non_dom_0'].to_numpy(), d['kpeaks_non_dom_1'].to_numpy()), -1)
        temporal_weights_non_dom = np.stack(
            (d['weight_non_dom_0'].to_numpy(), d['weight_non_dom_1'].to_numpy()), -1)
        temporal_delays_non_dom = np.stack(
            (d['delay_non_dom_0'].to_numpy(), d['delay_non_dom_1'].to_numpy()), -1)

        # values from bmtk
        t_path = os.path.join(root_path, f'temporal_kernels_{col_size}x{row_size}.pkl')
        kernel_length = 700
        if not os.path.exists(t_path):
            nkt = 600
            dom_temporal_kernels = []
            non_dom_temporal_kernels = []
            print('Computing temporal kernels')
            # for i in tqdm.tqdm(range(x.shape[0])):
            for i in range(x.shape[0]):
                dom_temporal_kernel = np.zeros((kernel_length,), np.float32)
                non_dom_temporal_kernel = np.zeros(
                    (kernel_length,), np.float32)
                if model_id[i].count('ON') > 0 and model_id[i].count('OFF') > 0:
                    non_dom_params = dict(
                        opt_wts=temporal_weights_non_dom[i],
                        opt_kpeaks=temporal_peaks_non_dom[i],
                        opt_delays=temporal_delays_non_dom[i]
                    )
                    dom_params = dict(
                        opt_wts=temporal_weights[i],
                        opt_kpeaks=temporal_peaks_dom[i],
                        opt_delays=temporal_delays[i]
                    )
                    amp_on = 1.0  # set the non-dominant subunit amplitude to unity

                    if model_id[i].count('sONsOFF_001') > 0:
                        non_dom_filter, non_dom_sum = create_one_unit_of_two_subunit_filter(
                            non_dom_params, 121.)
                        dom_filter, dom_sum = create_one_unit_of_two_subunit_filter(
                            dom_params, 115.)

                        spont = 4.0
                        max_roff = 35.0
                        max_ron = 21.0
                        amp_off = -(max_roff / max_ron) * (non_dom_sum / dom_sum) * amp_on - (
                            spont * (max_roff - max_ron)) / (max_ron * dom_sum)
                    elif model_id[i].count('sONtOFF_001') > 0:
                        non_dom_filter, non_dom_sum = create_one_unit_of_two_subunit_filter(
                            non_dom_params, 93.5)
                        dom_filter, dom_sum = create_one_unit_of_two_subunit_filter(
                            dom_params, 64.8)

                        spont = 5.5
                        max_roff = 46.0
                        max_ron = 31.0
                        amp_off = -0.7 * (max_roff / max_ron) * (non_dom_sum / dom_sum) * amp_on - (
                            spont * (max_roff - max_ron)) / (max_ron * dom_sum)
                    else:
                        raise ValueError('Unknown cell type')
                    non_dom_amplitude[i] = amp_on
                    amplitude[i] = amp_off
                    spontaneous_firing_rates[i] = spont / 2

                    hor_offset = np.cos(
                        tuning_angle[i] * np.pi / 180.) * subfield_separation[i] + x[i]
                    vert_offset = np.sin(
                        tuning_angle[i] * np.pi / 180.) * subfield_separation[i] + y[i]
                    non_dominant_x[i] = hor_offset
                    non_dominant_y[i] = vert_offset
                    dom_temporal_kernel[-len(dom_filter.kernel_data):] = dom_filter.kernel_data[::-1]
                    non_dom_temporal_kernel[-len(non_dom_filter.kernel_data):] = non_dom_filter.kernel_data[::-1]
                else:
                    dd = dict(neye=0, ncos=2, kpeaks=temporal_peaks_dom[i], b=.3,
                              delays=[temporal_delays[i].astype(int)])
                    kernel_data = np.dot(makeBasis_StimKernel(
                        dd, nkt), temporal_weights[i])
                    dom_temporal_kernel[-len(kernel_data):] = kernel_data

                dom_temporal_kernels.append(dom_temporal_kernel)
                non_dom_temporal_kernels.append(non_dom_temporal_kernel)
            dom_temporal_kernels = np.array(
                dom_temporal_kernels).astype(np.float32)
            non_dom_temporal_kernels = np.array(
                non_dom_temporal_kernels).astype(np.float32)
            to_save = dict(
                dom_temporal_kernels=dom_temporal_kernels,
                non_dom_temporal_kernels=non_dom_temporal_kernels,
                non_dominant_x=non_dominant_x,
                non_dominant_y=non_dominant_y,
                amplitude=amplitude,
                non_dom_amplitude=non_dom_amplitude,
                spontaneous_firing_rates=spontaneous_firing_rates
            )
            with open(t_path, 'wb') as f:
                pkl.dump(to_save, f)
                print('Caching temporal kernels...')
        else:
            with open(t_path, 'rb') as f:
                loaded = pkl.load(f)
            dom_temporal_kernels = loaded['dom_temporal_kernels']
            non_dom_temporal_kernels = loaded['non_dom_temporal_kernels']
            non_dominant_x = loaded['non_dominant_x']
            non_dominant_y = loaded['non_dominant_y']
            amplitude = loaded['amplitude']
            non_dom_amplitude = loaded['non_dom_amplitude']
            spontaneous_firing_rates = loaded['spontaneous_firing_rates']
        truncation = np.min(
            np.sum(np.cumsum(np.abs(dom_temporal_kernels), axis=1) <= 1e-6, 1))
        non_dom_truncation = np.min(
            np.sum(np.cumsum(np.abs(non_dom_temporal_kernels), axis=1) <= 1e-6, 1))
        truncation = np.min([truncation, non_dom_truncation])
        print(f'Could truncate {truncation} steps from filter')

        x = x * (col_size-1) / col_size  # 239 / 240
        y = y * (row_size-1) / row_size  # 119 / 120
        x[np.floor(x) < 0] = 0.
        y[np.floor(y) < 0] = 0.
        x[np.ceil(x) >= float(col_size-1)] = float(col_size-1)
        y[np.ceil(y) >= float(row_size-1)] = float(row_size-1)

        non_dominant_x = non_dominant_x * (col_size-1) / col_size  # 239 / 240
        non_dominant_y = non_dominant_y * (row_size-1) / row_size
        non_dominant_x[np.floor(non_dominant_x) < 0] = 0.
        non_dominant_y[np.floor(non_dominant_y) < 0] = 0.
        non_dominant_x[np.ceil(non_dominant_x) >= float(col_size-1)] = float(col_size-1)
        non_dominant_y[np.ceil(non_dominant_y) >= float(row_size-1)] = float(row_size-1)

        # prepare the spatial kernels in advance and store in TF format
        d_spatial = 1.
        spatial_range = np.arange(0, 15, d_spatial)
        x_range = np.arange(-50, 51)  # define the spatial kernel max size
        y_range = np.arange(-50, 51)

        # kernels = []
        gaussian_filters = []
        for i in range(len(spatial_range) - 1):
            sigma = np.round(np.mean(spatial_range[i:i+2])) / 3
            original_filter = GaussianSpatialFilter(translate=(
                0., 0.), sigma=(sigma, sigma), origin=(0., 0.))
            kernel = original_filter.get_kernel(
                x_range, y_range, amplitude=1.).full()
            # kernels.append(kernel)
            nonzero_inds = np.where(np.abs(kernel) > 1e-9)
            rm, rM = nonzero_inds[0].min(), nonzero_inds[0].max()
            cm, cM = nonzero_inds[1].min(), nonzero_inds[1].max()
            kernel = kernel[rm:rM + 1, cm:cM + 1]
            gaussian_filter = kernel[..., None, None]
            gaussian_filter = tf.convert_to_tensor(
                gaussian_filter, dtype=tf.float32)
            gaussian_filters.append(gaussian_filter)

        self.x = x
        self.y = y
        self.non_dominant_x = non_dominant_x
        self.non_dominant_y = non_dominant_y
        self.amplitude = amplitude
        self.non_dom_amplitude = non_dom_amplitude
        self.spontaneous_firing_rates = spontaneous_firing_rates
        self.dom_temporal_kernels = dom_temporal_kernels
        self.non_dom_temporal_kernels = non_dom_temporal_kernels
        # self.kernels = kernels
        self.gaussian_filters = gaussian_filters

    def spatial_response(self, movie):
        d_spatial = 1.
        spatial_range = np.arange(0, 15, d_spatial)

        x = self.x
        y = self.y
        non_dominant_x = self.non_dominant_x
        non_dominant_y = self.non_dominant_y
        spatial_sizes = self.spatial_sizes
        # x_range = np.arange(-50, 51)
        # y_range = np.arange(-50, 51)

        all_spatial_responses = []
        neuron_ids = []

        all_non_dom_spatial_responses = []

        for i in range(len(spatial_range) - 1):
            sel = np.logical_and(
                spatial_sizes < spatial_range[i + 1], spatial_sizes >= spatial_range[i])
            if np.sum(sel) <= 0:
                continue
            neuron_ids.extend(np.where(sel)[0])

            # construct spatial filter
            sigma = np.round(np.mean(spatial_range[i:i+2])) / 3.
            # original_filter = GaussianSpatialFilter(translate=(0., 0.), sigma=(sigma, sigma), origin=(0., 0.))
            # kernel = original_filter.get_kernel(x_range, y_range, amplitude=1.).full()
            # kernel = self.kernels[i]
            # nonzero_inds = np.where(np.abs(kernel) > 1e-9)
            # rm, rM = nonzero_inds[0].min(), nonzero_inds[0].max()
            # cm, cM = nonzero_inds[1].min(), nonzero_inds[1].max()
            # kernel = kernel[rm:rM + 1, cm:cM + 1]
            # gaussian_filter = kernel[..., None, None]
            gaussian_filter = self.gaussian_filters[i]

            # gaussian_filter = create_gaussian_filter(np.round(np.mean(spatial_range[i:i+2])))
            # apply it
            # convolved_movie = tf.nn.conv2d(movie, gaussian_filter, strides=[1, 1], padding='SAME')[..., 0]
            convolved_movie = self.do_conv(movie, gaussian_filter)
            # select items
            spatial_responses = select_spatial(x[sel], y[sel], convolved_movie)
            non_dom_spatial_responses = select_spatial(
                non_dominant_x[sel], non_dominant_y[sel], convolved_movie)
            all_spatial_responses.append(spatial_responses)
            all_non_dom_spatial_responses.append(non_dom_spatial_responses)
        neuron_ids = np.array(neuron_ids)
        all_spatial_responses = tf.concat(all_spatial_responses, 1)
        all_non_dom_spatial_responses = tf.concat(
            all_non_dom_spatial_responses, 1)

        sorted_neuron_ids_indices = np.argsort(neuron_ids)
        all_spatial_responses = tf.gather(
            all_spatial_responses, sorted_neuron_ids_indices, axis=1)
        all_non_dom_spatial_responses = tf.gather(
            all_non_dom_spatial_responses, sorted_neuron_ids_indices, axis=1)

        # print(f'Dominant spatial reponses computed: {all_spatial_responses.shape}')
        # print(f'Non dominant spatial reponses computed: {all_non_dom_spatial_responses.shape}')

        return all_spatial_responses, all_non_dom_spatial_responses
    # @tf.function(jit_compile=True)

    def do_conv(self, movie, gaussian_filter):
        return tf.nn.conv2d(movie, gaussian_filter, strides=[1, 1], padding='SAME')[..., 0]

    # @tf.function(jit_compile=True) # This works for TF 2.7.0 an onwards
    def firing_rates_from_spatial(self, all_spatial_responses, all_non_dom_spatial_responses):
        dom_filtered_output = temporal_filter(
            all_spatial_responses, self.dom_temporal_kernels)
        non_dom_filtered_output = temporal_filter(
            all_non_dom_spatial_responses, self.non_dom_temporal_kernels)
        # combined_filtered_output = dom_filtered_output * amplitude + non_dom_filtered_output * non_dom_amplitude
        firing_rates = transfer_function(
            dom_filtered_output * self.amplitude + self.spontaneous_firing_rates)
        multi_firing_rates = firing_rates + transfer_function(
            non_dom_filtered_output * self.non_dom_amplitude + self.spontaneous_firing_rates)
        firing_rates = firing_rates * \
            (1 - self.is_composite) + multi_firing_rates * self.is_composite
        return firing_rates


def main():
    from check_filter import load_example_movie
    movie = load_example_movie(duration=2000, onset=1000, offset=1100)

    lgn = LGN()
    spatial = lgn.spatial_response(movie)
    firing_rates = lgn.firing_rates_from_spatial(*spatial)

    # fig, ax = plt.subplots(figsize=(12, 12))
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(5, 1)
    ax = fig.add_subplot(gs[:4])
    if False:
        import h5py
        f = h5py.File(
            '/data/allen/v1_model/go_nogo_image_outputs/stim_0.h5_f_tot.h5', mode='r')
        d = np.array(f['firing_rates_Hz'])
        data = firing_rates[:, :4000].numpy().T - d[:4000]
        abs_max = np.abs(data).max()
        p = ax.pcolormesh(data, cmap='seismic', vmin=-abs_max, vmax=abs_max)
    else:
        data = firing_rates.numpy().T
        p = ax.pcolormesh(data, cmap='cividis')
    plt.colorbar(p, ax=ax)
    ax = fig.add_subplot(gs[4])
    ax.plot(data.mean(0))
    fig.savefig('temp.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
