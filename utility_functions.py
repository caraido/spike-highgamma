import mat73
import os
import pandas as pd
import scipy.io as sio
import numpy as np
import math
from itertools import chain
import yaml
import matplotlib as mpb
import pickle as pk

DATA_PATH='./data'
data_heads=['file_num','file_types','num_success','num_failed','success_rate','spike_rate','HGA','target_types','velocity','position','trial_types']

def load_ONF_data(data_type: str, monkey: str, electrode_number: int) -> pd.DataFrame:
    """Load .mat data file for specified monkey and electrode number.

    Args:
        data_type: Type of data to load ('last4s' 'trials')
        monkey: Monkey identifier string
        electrode_number: Electrode number to load. Note this is not electrode index

    Returns:
        data from raw data

    Raises:
        FileNotFoundError: If matching .mat file is not found
    """
    data_path = os.path.join(DATA_PATH, data_type)
    assert os.path.isdir(data_path)
    files = os.listdir(data_path)

    target_file = f"{monkey}{electrode_number}.mat"
    if target_file not in files:
        raise FileNotFoundError(f"Could not find file {target_file} in {data_path}")

    file_path = os.path.join(data_path, target_file)

    # load .mat file
    raw_data=mat73.loadmat(file_path)['everything']
    # convert to pandas dataframe
    raw_data=pd.DataFrame(raw_data, columns=data_heads)
    return raw_data

def load_STA_data(data_type: str, monkey: str, electrode_number: int) -> pd.DataFrame:
    data_path = os.path.join(DATA_PATH, data_type)
    assert os.path.isdir(data_path)
    files = os.listdir(data_path)

    target_file = f"{monkey}{electrode_number}.pk"
    if target_file not in files:
        raise FileNotFoundError(f"Could not find file {target_file} in {data_path}")
    file_path = os.path.join(data_path, target_file)

    # load .mat file
    with open(file_path, 'rb') as f:
        raw_data = pk.load(f)
    # convert to pandas dataframe
    raw_data = pd.DataFrame(raw_data)
    return raw_data

def load_shunted_electrodes(monkey:str,spike_channel):
    """Load shunted electrodes for specified monkey"""
    shunted= sio.loadmat(os.path.join(DATA_PATH, 'shunted_electrodes.mat'))
    shunted_electrodes = shunted[monkey][0]-1
    if spike_channel-1 in shunted_electrodes:
        shunted_electrodes=shunted_electrodes[shunted_electrodes!=spike_channel-1]
    return shunted_electrodes # this is python index

def file_2_day(df, column, axis):
    """
    Concatenate files in a DataFrame column grouped by 'date'.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing a 'date' column and array column.
        column (str): Name of the column with arrays to concatenate.
        axis (int): Axis along which to concatenate.

    Returns:
        np.ndarray: Concatenated array across all dates.
    """
    grouped_arrays = []

    for _, group in df.groupby('date'):
        arrays = group[column].values  # list of np.ndarrays
        try:
            stacked = np.concatenate(arrays, axis=axis)
        except:
            stacked = list(chain(*arrays))

        grouped_arrays.append(stacked)

    return grouped_arrays

def closest_factors_decomposition(n):
    # This is mostly for plotting. It arranges the subplots as close# as a square
    # Start from the square root of the number and go downwards
    for i in range(math.isqrt(n), 0, -1):
        if n % i == 0:
            # Found the two closest factors
            return i, n // i, 0
        elif n % i < i:
            # Found the closest factors with the smallest residual
            return i, (n // i) + 1, (i * ((n // i) + 1)) - n
        return None
    return None

def get_CE_loc(grid, channel):
    loc_x_sp, loc_y_sp = np.where(grid == channel)
    loc_x = loc_x_sp[0]
    loc_y = loc_y_sp[0]
    return loc_x, loc_y

def get_channel(monkey, channel):
    # get spike channel. This is decided by the wiring of the electrodes and the jackbox
    if monkey == 'Mini':
        spike_channel = channel - 32
    elif monkey == 'Jaco':
        spike_channel = channel + 64 if channel + 64 <= 96 else channel + 64 - 96
    else:  # for Chewie
        spike_channel = channel

    return spike_channel


    # get the channel index
def load_pin_map(monkey):
    metadata_path=DATA_PATH+'/metadata'
    with open(os.path.join(metadata_path, 'mapping.yaml'),'r') as f:
        mapping=yaml.safe_load(f)
    return np.array(mapping[monkey]['pin_map'])

def load_electrode_map(monkey):
    metadata_path=DATA_PATH+'/metadata'
    with open(os.path.join(metadata_path, 'mapping.yaml'),'r') as f:
        mapping=yaml.safe_load(f)
    return np.array(mapping[monkey]['electrode_map'])

def get_grid(monkey):
    pin_map=load_pin_map(monkey)
    electrode_map=load_electrode_map(monkey)
    if monkey == 'M' or monkey == 'C':
        all_combines = np.array([pin_map, electrode_map, pin_map])
        sorted_all = all_combines.T[all_combines[1, :].argsort()].T  # sort based on the monkey map
        grid = np.flipud(np.reshape(sorted_all[2, :], [10, 10]))

    else:
        grid = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0,
                         9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                         19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                         29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                         39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                         49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                         59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
                         69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
                         79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                         0, 89, 90, 91, 92, 93, 94, 95, 96, 0
                         ]).reshape(10, 10)
        grid = np.fliplr(np.flipud(grid.T))
    return grid


def plot_on_grid(ax,
                 leftover_electrode_list,grid,
                 data,
                 loc=None,
                 cmap=mpb.cm.jet,
                 show_chan_indx=False,
                 CE_color='g',
                 CE_linewidth=4 ):

    full_electrode = np.empty(96)
    full_electrode[:] = np.nan
    full_electrode[leftover_electrode_list] = data
    elec = []

    for i in range(100):
        if grid.flatten()[i]:  # or i!=0 or i!=9 or i!=90 or i!=99:
            elec.append(full_electrode[grid.flatten()[i] - 1])  # turn in to python index
        else:
            elec.append(np.NaN)
    elec = np.reshape(elec, [10, 10])
    cmap.set_bad('white',1.)

    im=ax.imshow(elec,cmap=cmap)
    if show_chan_indx:
        for (k, j), label in np.ndenumerate(grid):
            ax.text(j, k, label, ha='center', va='center')
    if loc is not None:
        assert len(loc)==2
        ax.add_patch(mpb.patches.Rectangle((loc[1]-0.5,loc[0]-0.5),1,1,fc='none',ec=CE_color,lw=CE_linewidth))
    ax.set_xticks([])
    ax.set_yticks([])
    return im

def draw_random_pairs(n_random_pairs,except_loc):
    '''
    This function draws the coordinate of random electrodes in a grid, but will avoid drawing from "except_loc"
    :param n_random_pairs: number of random electrodes to draw
    :param except_loc: the electrode coordinate that is excluded
    :return:
    '''
    random_pairs=[]
    pair=0
    corner=[np.array([0,0]),np.array([0,9]),np.array([9,0]),np.array([9,9])]
    while pair<n_random_pairs:
        this_pair=np.random.choice(9,2)
        if len(random_pairs)==0 and not np.any(np.all(this_pair==corner,axis=1)):
            random_pairs.append(this_pair)
            pair+=1
            continue
        if not np.any(np.all(this_pair==random_pairs,axis=1)) and not np.any(np.all(this_pair==corner,axis=1)) and not np.all(this_pair==np.array(except_loc)):
            random_pairs.append(this_pair)
            pair+=1
    return random_pairs

def get_the_weights(decay_grid,leftover_electrode_list,flat_grid):
    weights=decay_grid.reshape(-1)
    weights=weights[[i-1 in leftover_electrode_list for i in flat_grid]]
    flat_order=np.argsort(flat_grid[[i-1 in leftover_electrode_list for i in flat_grid]])
    weights=weights[flat_order]
    return weights

def calculate_weighted_grid(start_location, power=2,
                     grid_size=10,
                     ce_distance=100,
                     electrode_distance=400,
                     emit_start_location=False):
    """
    Calculate the values for each electrode in a grid based on the 1/d^(power) decay rule,
    where d is the Euclidean distance from the start location.

    :param start_location: Tuple (x, y) representing the starting location on the grid
    :param grid_size: Size of the grid, default is 10x10
    :param ce_distance: the maximum hearing distance of the electrode
    :param electrode_distance: the distance between two electrodes
    :return: A grid_size x grid_size array with calculated values
    """
    values = np.zeros((grid_size, grid_size))
    # everything in um.
    for i in range(grid_size):
        for j in range(grid_size):
            unit_distance = np.sqrt((start_location[0] - i) ** 2 + (start_location[1] - j) ** 2)
            if emit_start_location:
                if unit_distance == 0:
                    values[i, j] = 0
                else:
                    values[i, j] = 1 / (unit_distance * electrode_distance) ** power
            else:
                if unit_distance == 0:
                    values[i, j] = 1 / ce_distance ** power
                else:
                    values[i, j] = 1 / (unit_distance * electrode_distance) ** power

    return values


## The following 3 functions are for cPCA
def get_cpca_loadings(fg_cov,bg_cov,alpha,n_components_cpca):
    sigma = fg_cov - alpha*bg_cov
    w, v = np.linalg.eig(sigma)
    eig_idx = np.argpartition(w, -n_components_cpca)[-n_components_cpca:]
    eig_idx = eig_idx[np.argsort(-w[eig_idx])]
    v_top = v[:,eig_idx]
    return np.real(v_top)

def reduce_dataset(dataset,alpha,fg_cov,bg_cov):
    v_top=get_cpca_loadings(fg_cov,bg_cov,alpha)
    reduced_dataset = dataset.dot(v_top)
    reduced_dataset[:,0] = reduced_dataset[:,0]*np.sign(reduced_dataset[0,0])
    reduced_dataset[:,1] = reduced_dataset[:,1]*np.sign(reduced_dataset[0,1])
    return reduced_dataset

def get_cov(fg,bg):
    bg_cov = bg.T.dot(bg)/(bg.shape[0]-1)
    fg_cov = fg.T.dot(fg)/(fg.shape[0]-1)
    return fg_cov,bg_cov

def get_the_star(p):
    # input p value and output the stars or n.s. to indicate significance
    if p < 0.005:
        sig_symbol = '***'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    elif p > 0.05:
        sig_symbol='n.s.'
    return sig_symbol


