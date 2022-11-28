"""Create Trajnet data from original datasets."""
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

import pysparkling
import scipy.io
import json
from . import readers
from .scene import Scenes
from .get_type import trajectory_type

from l5kit.data import ChunkedDataset, LocalDataManager
from tqdm import tqdm
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points
from l5kit.configs import load_config_data
from l5kit.dataset import EgoDataset, AgentDataset

import warnings
warnings.filterwarnings("ignore")
matplotlib.use("QtAgg")

def biwi(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.biwi)
            .cache())


def crowds(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .values()
            .flatMap(readers.crowds)
            .cache())


def mot(sc, input_file):
    """Was 7 frames per second in original recording."""
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.mot)
            .filter(lambda r: r.frame % 2 == 0)
            .cache())


def edinburgh(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .zipWithIndex()
            .flatMap(readers.edinburgh)
            .cache())


def syi(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .flatMap(readers.syi)
            .cache())


def dukemtmc(sc, input_file):
    print('processing ' + input_file)
    contents = scipy.io.loadmat(input_file)['trainData']
    return (sc
            .parallelize(readers.dukemtmc(contents))
            .cache())


def wildtrack(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .flatMap(readers.wildtrack)
            .cache())

def cff(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.cff)
            .filter(lambda r: r is not None)
            .cache())

def lcas(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.lcas)
            .cache())

def controlled(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.controlled)
            .cache())

def get_trackrows(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.get_trackrows)
            .filter(lambda r: r is not None)
            .cache())

def standard(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.standard)
            .cache())

def car_data(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .flatMap(readers.car_data)
            .cache())

def woven_planet(sc, input_file):
    cfg = load_config_data("./visualisation_config.yaml")

    print('processing ' + input_file)
    dm = LocalDataManager()
    dataset_path = dm.require("sample.zarr")
    zarr_dataset = ChunkedDataset(dataset_path).open()
    # rast = build_rasterizer(cfg, dm)
    # dataset = AgentDataset(cfg, zarr_dataset, rast)
    # print(dataset.agent_from_world)
    # for agent in dataset:
    # for idx in range(11):
    scenes = []
    tracks = []
    last_idx = 0
    with open('output_pre/train/woven_planet.ndjson', 'w') as f:
        for scene_idx in range(4):
            start = zarr_dataset.scenes[scene_idx]["frame_index_interval"][0]
            end = zarr_dataset.scenes[scene_idx]["frame_index_interval"][1]
            host = zarr_dataset.scenes[scene_idx]["host"]
            frames = zarr_dataset.frames[start:end]
            line = {"scene": {"id": int(scene_idx), "p": 1, "s": int(start), "e": int(end), "fps": 2.5, "tag": 0}}
            
            scenes.append(line)
            for frame_idx, frame in enumerate(frames):
                start_agent = frame["agent_index_interval"][0]
                end_agent = frame["agent_index_interval"][1]
                agents = zarr_dataset.agents[start_agent:end_agent][zarr_dataset.agents[start_agent:end_agent]["label_probabilities"][:, 3] > 0.95]
                for agent in agents:
                    track = {"track": {"f": int(frame_idx) + last_idx, "p": int(agent["track_id"]), "x": float(agent["centroid"][0]), "y": float(agent["centroid"][1])}}
                    tracks.append(track)
                    
            last_idx = int(end) + last_idx
        
        for scene in scenes:
            f.write(json.dumps(scene) + "\n")
        for track in tracks:
            f.write(json.dumps(track) + "\n")

        # for frame_idx in range(1000):
        #     start = zarr_dataset.frames[frame_idx]["agent_index_interval"][0]
        #     end = zarr_dataset.frames[frame_idx]["agent_index_interval"][1]
        #     # agents = zarr_dataset.agents[idx]["agent_index_intervall"]
        #     agents = zarr_dataset.agents[start:end][zarr_dataset.agents[start:end]["label_probabilities"][:, 3] > 0.75]
        #     for agent in agents:
        #         f.write(str(frame_idx) + "\t" + str(agent["track_id"]) + "\t")
        #         f.write(str(agent["centroid"][0]) + "\t" + str(agent["centroid"][1])  + "\t")
        #         f.write('\n')







        # centroids.append(centroid)
    #     print(agent["centroid"])
    # with open('readme.txt', 'w') as f:
    #     for frame_idx in tqdm(range(1000)):
    #         data = dataset.get_frame_indices(frame_idx)
    #         if len(data) != 0:
    #             # print(data):
    #             for agent in data:
    #                 # print(dataset[agent]["label_probabilities"])
    #                 # print(dataset[agent]["centroid"])
    #                 print(dataset[agent]["agent_from_world"])
    #                 f.write(str(frame_idx) + "\t" + str(dataset[agent]["track_id"]) + "\t")
    #                 pos = transform_points(dataset[agent]["centroid"], data["agent_from_world"])
    #                 f.write(str(pos[0]) + "\t" + str(dpos[1])  + "\t")
    #                 f.write('\n')

        # for scene_idx in tqdm(range(3)):
        # data = dataset.get_scene_indices(3)
        # if len(data) != 0:
        #     # print(data):
        #     for idx in data:
        #         # print(dataset[agent]["timestamp"])
        #         # print(dataset[agent]["track_id"])
        #         f.write(str(dataset[agent]["timestamp"]) + "\t" + str(dataset[agent]["track_id"]) + "\t")
        #         f.write(str(dataset[agent]["centroid"][0]) + "\t" + str(dataset[agent]["centroid"][1])  + "\t")
        #         f.write('\n')

            # f.write(str(idx_data) + "\t" + str(random.randrange(0, 50)) + "\t")
            # f.write(str(frame["ego_translation"][:2][0]) + "\t" + str(frame["ego_translation"][:2][1]) + "\t")
            # f.write('\n')


    # return standard(sc, "/home/janos/projects/mpfav/trajnetplusplus-for-vehicles/trajnet++/trajnetplusplusdataset/readme.txt")          


def write(input_rows, output_file, args):
    """ Write Valid Scenes without categorization """

    print(" Entering Writing ")
    ## To handle two different time stamps 7:00 and 17:00 of cff
    if args.order_frames:
        frames = sorted(set(input_rows.map(lambda r: r.frame).toLocalIterator()),
                        key=lambda frame: frame % 100000)
    else:
        frames = sorted(set(input_rows.map(lambda r: r.frame).toLocalIterator()))

    # split
    train_split_index = int(len(frames) * args.train_fraction)
    val_split_index = train_split_index + int(len(frames) * args.val_fraction)
    train_frames = set(frames[:train_split_index])
    val_frames = set(frames[train_split_index:val_split_index])
    test_frames = set(frames[val_split_index:])

    # train dataset
    train_rows = input_rows.filter(lambda r: r.frame in train_frames)
    train_output = output_file.format(split='train')
    train_scenes = Scenes(fps=args.fps, start_scene_id=0, args=args).rows_to_file(train_rows, train_output)

    # validation dataset
    val_rows = input_rows.filter(lambda r: r.frame in val_frames)
    val_output = output_file.format(split='val')
    val_scenes = Scenes(fps=args.fps, start_scene_id=train_scenes.scene_id, args=args).rows_to_file(val_rows, val_output)

    # public test dataset
    test_rows = input_rows.filter(lambda r: r.frame in test_frames)
    test_output = output_file.format(split='test')
    test_scenes = Scenes(fps=args.fps, start_scene_id=val_scenes.scene_id, args=args) # !!! Chunk Stride
    test_scenes.rows_to_file(test_rows, test_output)

    # private test dataset
    private_test_output = output_file.format(split='test_private')
    private_test_scenes = Scenes(fps=args.fps, start_scene_id=val_scenes.scene_id, args=args)
    private_test_scenes.rows_to_file(test_rows, private_test_output)

def categorize(sc, input_file, args):
    """ Categorize the Scenes """

    print(" Entering Categorizing ")
    test_fraction = 1 - args.train_fraction - args.val_fraction

    train_id = 0
    if args.train_fraction:
        print("Categorizing Training Set")
        train_rows = get_trackrows(sc, input_file.replace('split', '').format('train'))
        train_id = trajectory_type(train_rows, input_file.replace('split', '').format('train'),
                                   fps=args.fps, track_id=0, args=args)

    val_id = train_id
    if args.val_fraction:
        print("Categorizing Validation Set")
        val_rows = get_trackrows(sc, input_file.replace('split', '').format('val'))
        val_id = trajectory_type(val_rows, input_file.replace('split', '').format('val'),
                                 fps=args.fps, track_id=train_id, args=args)


    if test_fraction:
        print("Categorizing Test Set")
        test_rows = get_trackrows(sc, input_file.replace('split', '').format('test_private'))
        _ = trajectory_type(test_rows, input_file.replace('split', '').format('test_private'),
                            fps=args.fps, track_id=val_id, args=args)

def edit_goal_file(old_filename, new_filename):
    """ Rename goal files. 
    The name of goal files should be identical to the data files
    """

    shutil.copy("goal_files/train/" + old_filename, "goal_files/train/" + new_filename)
    shutil.copy("goal_files/val/" + old_filename, "goal_files/val/" + new_filename)
    shutil.copy("goal_files/test_private/" + old_filename, "goal_files/test_private/" + new_filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_len', type=int, default=9,
                        help='Length of observation')
    parser.add_argument('--pred_len', type=int, default=12,
                        help='Length of prediction')
    parser.add_argument('--train_fraction', default=0.6, type=float,
                        help='Training set fraction')
    parser.add_argument('--val_fraction', default=0.2, type=float,
                        help='Validation set fraction')
    parser.add_argument('--fps', default=2.5, type=float,
                        help='fps')
    parser.add_argument('--order_frames', action='store_true',
                        help='For CFF')
    parser.add_argument('--chunk_stride', type=int, default=2,
                        help='Sampling Stride')
    parser.add_argument('--min_length', default=0.0, type=float,
                        help='Min Length of Primary Trajectory')
    parser.add_argument('--synthetic', action='store_true',
                        help='convert synthetic datasets (if false, convert real)')
    parser.add_argument('--direct', action='store_true',
                        help='directy convert synthetic datasets using commandline')
    parser.add_argument('--all_present', action='store_true',
                        help='filter scenes where all pedestrians present at all times')
    parser.add_argument('--orca_file', default=None,
                        help='Txt file for ORCA trajectories, required in direct mode')
    parser.add_argument('--goal_file', default=None,
                        help='Pkl file for goals (required for ORCA sensitive scene filtering)')
    parser.add_argument('--output_filename', default=None,
                        help='name of the output dataset filename constructed in .ndjson format, required in direct mode')
    parser.add_argument('--mode', default='default', choices=('default', 'trajnet'),
                        help='mode of ORCA scene generation (required for ORCA sensitive scene filtering)')

    ## For Trajectory categorizing and filtering
    categorizers = parser.add_argument_group('categorizers')
    categorizers.add_argument('--static_threshold', type=float, default=1.0,
                              help='Type I static threshold')
    categorizers.add_argument('--linear_threshold', type=float, default=0.5,
                              help='Type II linear threshold (0.3 for Synthetic)')
    categorizers.add_argument('--inter_dist_thresh', type=float, default=5,
                              help='Type IIId distance threshold for cone')
    categorizers.add_argument('--inter_pos_range', type=float, default=15,
                              help='Type IIId angle threshold for cone (degrees)')
    categorizers.add_argument('--grp_dist_thresh', type=float, default=0.8,
                              help='Type IIIc distance threshold for group')
    categorizers.add_argument('--grp_std_thresh', type=float, default=0.2,
                              help='Type IIIc std deviation for group')
    categorizers.add_argument('--acceptance', nargs='+', type=float, default=[0.1, 1, 1, 1],
                              help='acceptance ratio of different trajectory (I, II, III, IV) types')

    args = parser.parse_args()
    # Set Seed
    random.seed(42)
    np.random.seed(42)

    sc = pysparkling.Context()

    # Real datasets conversion
    if not args.synthetic:
        # write(standard(sc, 'data/eth/train/biwi_hotel_train.txt'),
        #     'output_pre/{split}/biwi_hotel_train.ndjson', args)
        # categorize(sc, 'output_pre/{split}/biwi_hotel_train.ndjson', args)
        # categorize(sc, 'output_pre/{split}/biwi_hotel_train.ndjson', args)
        # write(crowds(sc, 'data/raw/crowds/crowds_zara01.vsp'),
        #       'output_pre/{split}/crowds_zara01.ndjson', args)
        # categorize(sc, 'output_pre/{split}/crowds_zara01.ndjson', args)
        # write(crowds(sc, 'data/raw/crowds/crowds_zara03.vsp'),
        #       'output_pre/{split}/crowds_zara03.ndjson', args)
        # categorize(sc, 'output_pre/{split}/crowds_zara03.ndjson', args)
        # write(crowds(sc, 'data/raw/crowds/students001.vsp'),
        #       'output_pre/{split}/crowds_students001.ndjson', args)
        # categorize(sc, 'output_pre/{split}/crowds_students001.ndjson', args)
        # write(crowds(sc, 'data/raw/crowds/students003.vsp'),
        #       'output_pre/{split}/crowds_students003.ndjson', args)
        # categorize(sc, 'output_pre/{split}/crowds_students003.ndjson', args)

        # write(woven_planet(sc, 'd/home/janos/projects/mpfav/trajnetplusplus-for-vehicles/trajnet++/trajnetplusplusdataset/data/sample.zarr'),'output_pre/{split}/woven_planet.ndjson', args)
        # # # new datasets
        woven_planet(sc, 'd/home/janos/projects/mpfav/trajnetplusplus-for-vehicles/trajnet++/trajnetplusplusdataset/data/sample.zarr')
        categorize(sc, 'output_pre/{split}/woven_planet.ndjson', args)

        # args.fps = 2
        # write(wildtrack(sc, 'data/raw/wildtrack/Wildtrack_dataset/annotations_positions/*.json'),
        #       'output_pre/{split}/wildtrack.ndjson', args)
        # categorize(sc, 'output_pre/{split}/wildtrack.ndjson', args)
        # args.fps = 2.5 # (Default)

        # # CFF: More trajectories
        # # Chunk_stride > 20 preferred & order_frames.
        # args.chunk_stride = 20
        # args.order_frames = True
        # write(cff(sc, 'data/raw/cff_dataset/al_position2013-02-06.csv'),
        #       'output_pre/{split}/cff_06.ndjson', args)
        # categorize(sc, 'output_pre/{split}/cff_06.ndjson', args)
        # args.chunk_stride = 2 # (Default)
        # args.order_frames = False # (Default)

    # Direct synthetic datasets conversion
    elif args.direct:
        # Note: Generate Trajectories First! See command below
        ## 'python -m trajnetdataset.controlled_data <args>'
        print("Direct Synthetic Data Converion")
        assert args.orca_file is not None
        assert args.goal_file is not None
        assert args.output_filename is not None
        write(controlled(sc, args.orca_file), 'output_pre/{split}/' + f'{args.output_filename}.ndjson', args)
        categorize(sc, 'output_pre/{split}/' + f'{args.output_filename}.ndjson', args)
        edit_goal_file(args.goal_file.split('/')[-1], f'{args.output_filename}.pkl')

    # Manual synthetic datasets conversion
    else:
        # Note: Generate Trajectories First! See command below
        ## 'python -m trajnetdataset.controlled_data <args>'
        print("Manual Synthetic Data Converion")
        write(controlled(sc, 'data/raw/controlled/orca_circle_crossing_5ped_1000scenes_.txt'),
              'output_pre/{split}/orca_five_synth.ndjson', args)
        categorize(sc, 'output_pre/{split}/orca_five_synth.ndjson', args)
        edit_goal_file('orca_circle_crossing_5ped_1000scenes_.pkl', 'orca_five_synth.pkl')

if __name__ == '__main__':
    main()
