# Test sets for Oxford dataset.

import argparse
from typing import List
import os
import cv2
import tqdm
from prnet.datasets.oxford.oxford_raw import OxfordSequence
from prnet.datasets.base_datasets import EvaluationTuple, EvaluationSet
from prnet.datasets.dataset_utils import filter_query_elements
from prnet.datasets.panorama import generate_sph_image


def load_img_file_oxford(filename, cam_mode):
    input_image = cv2.imread(filename)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    return input_image


def get_scans(sequence: OxfordSequence) -> List[EvaluationTuple]:
    # Get a list of all readings from the test area in the sequence
    elems = []
    for ndx in range(len(sequence)):
        pose = sequence.poses[ndx]
        position = pose[:2, 3]
        item = EvaluationTuple(sequence.timestamps[ndx], sequence.rel_scan_filepath[ndx], position=position, pose=pose, filepaths=sequence.filepaths[ndx])
        elems.append(item)
    return elems


def generate_evaluation_set(dataset_root: str, map_sequence: str, query_sequence: str, sampling_distance: float = 0.2,
                            dist_threshold=20, sph: bool = False) -> EvaluationSet:
    split = 'test'
    map_sequence = OxfordSequence(dataset_root, map_sequence, split=split, sampling_distance=sampling_distance)
    query_sequence = OxfordSequence(dataset_root, query_sequence, split=split, sampling_distance=sampling_distance)

    map_set = get_scans(map_sequence)
    query_set = get_scans(query_sequence)

    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    query_set = filter_query_elements(query_set, map_set, dist_threshold)

    for i in tqdm.tqdm(range(len(query_set))):
        file_pathname = query_set[i].filepaths
        camera_mode = ["mono_left", "mono_right", "mono_rear", "stereo"]
        images = [load_img_file_oxford(file_pathname[i], camera_mode[i-2]) for i in range(2, 6)]
        sph_filename = file_pathname[2].replace('png', 'png')
        sph_filename = sph_filename.replace('mono_left_rect', 'sph')
        sph_img = generate_sph_image(images, 'oxford', "/media/workspace/dataset/Oxford/image_meta.pkl", dataset_root)
        cv2.imwrite(sph_filename, sph_img)

    for i in tqdm.tqdm(range(len(map_set))):
        file_pathname = map_set[i].filepaths
        camera_mode = ["mono_left", "mono_right", "mono_rear", "stereo"]
        images = [load_img_file_oxford(file_pathname[i], camera_mode[i-2]) for i in range(2, 6)]
        sph_filename = file_pathname[2].replace('png', 'png')
        sph_filename = sph_filename.replace('mono_left_rect', 'sph')
        sph_img = generate_sph_image(images, 'oxford', "/media/workspace/dataset/Oxford/image_meta.pkl", dataset_root)
        cv2.imwrite(sph_filename, sph_img)

    print(f'{len(map_set)} database elements, {len(query_set)} query elements')
    return EvaluationSet(query_set, map_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for Oxford dataset')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--sampling_distance', type=float, default=0.2)
    # Ignore query elements that do not have a corresponding map element within the given threshold (in meters)
    parser.add_argument('--dist_threshold', type=float, default=2.0)
    parser.add_argument('--sph', type=bool, default=False)
    args = parser.parse_args()

    print(f'Dataset root: {args.dataset_root}')
    print(f'Minimum displacement between consecutive anchors: {args.sampling_distance}')
    print(f'Ignore query elements without a corresponding map element within a threshold [m]: {args.dist_threshold}')

    # Sequences is a list of (map sequence, query sequence)
    sequences = [('2019-01-11-13-24-51-radar-oxford-10k', '2019-01-15-13-06-37-radar-oxford-10k')]

    for map_sequence, query_sequence in sequences:
        print(f'Map sequence: {map_sequence}')
        print(f'Query sequence: {query_sequence}')

        test_set = generate_evaluation_set(args.dataset_root, map_sequence, query_sequence,
                                           sampling_distance=args.sampling_distance, dist_threshold=args.dist_threshold, sph=args.sph)

        pickle_name = f'test_{map_sequence}_{query_sequence}_{args.sampling_distance}.pickle'
        file_path_name = os.path.join(args.dataset_root, pickle_name)
        test_set.save(file_path_name)
