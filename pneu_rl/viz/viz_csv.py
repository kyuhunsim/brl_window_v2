import os
import sys

from typing import Dict
from collections import deque
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

############################################################
def plot(
    data: Dict[str, deque],
    fig_name: str = 'fig'
):
    color = deque(['red', 'blue', 'green'])
    fontname = 'Times New Roman'
    label_font_size = 18
    
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(5, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0:2,0])
    ax2 = fig.add_subplot(gs[2:4,0])
    ax3 = fig.add_subplot(gs[4,0])    

    key = 'real.csv'
    ax1.plot(
        datas[key]['time'],
        datas[key]['ref_pos'],
        linewidth=2, color='black', linestyle='dashed', label='REF'
    )
    ax1.plot(
        datas[key]['time'],
        datas[key]['sen_pos'],
        linewidth=2, color='red', label='POS'
    )
    ax1.set_ylabel('Pressure [kPa]', fontname=fontname, fontsize=label_font_size)
    ax1.grid(which='major', color='silver', linewidth=1)
    ax1.grid(which='minor', color='lightgray', linewidth=0.5)
    ax1.minorticks_on()
    ax1.legend(loc='upper right')
    ax1.set(xlim=(None, None), ylim=(None, None))
    
    ax2.plot(
        datas[key]['time'],
        datas[key]['ref_neg'],
        linewidth=2, color='black', linestyle='dashed', label='REF'
    )
    ax2.plot(
        datas[key]['time'],
        datas[key]['sen_neg'],
        linewidth=2, color='blue', label='NEG'

    )
    ax2.set_ylabel('Pressure [kPa]', fontname=fontname, fontsize=label_font_size)
    ax2.grid(True)
    ax2.grid(which='major', color='silver', linewidth=1)
    ax2.grid(which='minor', color='lightgray', linewidth=0.5)
    ax2.minorticks_on()
    ax2.legend(loc='upper right')
    ax2.sharex(ax1)
    ax2.set(xlim=(None, None), ylim=(None, None))

    ax3.plot(
        datas[key]['time'],
        datas[key]['ctrl_pos'],
        linewidth=2, color='red', label='POS'
    )
    ax3.plot(
        datas[key]['time'],
        datas[key]['ctrl_neg'],
        linewidth=2, color='blue', label='NEG'

    )
    ax3.set_xlabel('Time [sec]', fontname=fontname, fontsize=label_font_size)
    ax3.set_ylabel('Control', fontname=fontname, fontsize=label_font_size)
    ax3.legend(loc='upper right')
    ax3.sharex(ax1)
    ax3.set(xlim=(None, None), ylim=(None, None))

    plt.tight_layout()
    plt.savefig(f'{fig_name}.png')
    print(f'[ INFO] Visualize ==> Saved file name: {fig_name}.png')
    plt.show()
    
    
############################################################
def get_bag_file_name():
    print(color('[INPUT] Visualize file:', 'blue'))
    bags = os.listdir('./bag')
    for i, bag in enumerate(bags):
        print(color(f'\t{i+1}. {bag}', 'yellow'))
    print(color('\t---', 'blue'))
    bag_idx = int(input(color('\tMODEL: ', 'blue'))) - 1 
    bag_name = bags[bag_idx]
    delete_lines(len(bags) + 3)
    print(f'[ INFO] Visualize file: {bag_name}')

    return bag_name[0:-4]
    
def get_csv_file_name():
    print(color('[INPUT] Visualize file:', 'blue'))
    csvs = os.listdir('./csv')
    for i, csv in enumerate(csvs):
        print(color(f'\t{i+1}. {csv}', 'yellow'))
    print(color('\t---', 'blue'))
    csv_idx = int(input(color('\tMODEL: ', 'blue'))) - 1 
    csv_name = csvs[csv_idx]
    delete_lines(len(csvs) + 3)
    print(f'[ INFO] Visualize file: {csv_name}')

    return csv_name[0:-4]

def get_datas_from_bag(
    bag_file: str,
    # topic: str = '/obs_wo_noise'
    topic: str = '/obs'
):
    del bag_file
    del topic
    raise NotImplementedError(
        "Bag parsing is removed in the ROS-free codebase. Use get_datas_from_csv()."
    )

def get_datas_from_csv(
    csv_file: str,
    topic: str = '/obs'
):
    csv = pd.read_csv(csv_file)

    topic_data = dict(
            time = deque(csv[csv.columns[1]]),
            sen_pos = deque(csv[csv.columns[2]]),
            sen_neg = deque(csv[csv.columns[3]]),
            ref_pos = deque(csv[csv.columns[4]]),
            ref_neg = deque(csv[csv.columns[5]]),
            ctrl_pos = deque(csv[csv.columns[6]]),
            ctrl_neg = deque(csv[csv.columns[7]])
        )
    return topic_data

def delete_lines(
    num: int
):
    for _ in range(num):
        sys.stdout.write("\033[F")  # Move the cursor up one line
        sys.stdout.write("\033[K")  # Clear the line

def color(
    line: str,
    color: str
):
    if color == 'blue':
        return '\033[94m' + f'{line}' + '\033[0m'
    elif color == 'yellow':
        return '\033[33m' + f'{line}' + '\033[0m'
    else:
        return line



if __name__ == '__main__':
    csv_name = 'real.csv'

    datas = {}
    datas[f'{csv_name}'] = get_datas_from_csv(csv_name)
    
    plot(datas)
