# Test sets for NCLT dataset.

import argparse
from typing import List
import os
import tqdm
from prnet.datasets.nclt.nclt_raw import NCLTSequence, load_im_file_for_generate, pc2image_file
from prnet.datasets.base_datasets import EvaluationTuple, EvaluationSet
from prnet.datasets.dataset_utils import filter_query_elements
from prnet.datasets.panorama import generate_sph_image
import cv2

DEBUG = False

def get_scans(sequence: NCLTSequence) -> List[EvaluationTuple]:
    # Get a list of all readings from the test area in the sequence
    elems = []
    for ndx in range(len(sequence)):
        pose = sequence.poses[ndx]
        position = pose[:2, 3]
        item = EvaluationTuple(sequence.timestamps[ndx], sequence.rel_scan_filepath[ndx], position=position, pose=pose)
        elems.append(item)
    return elems


def generate_evaluation_set(dataset_root: str, map_sequence: str, query_sequence: str, sampling_distance: float = 0.2,
                            dist_threshold=20, sph: bool = False) -> EvaluationSet:
    split = 'test'
    map_sequence = NCLTSequence(dataset_root, map_sequence, split=split, sampling_distance=sampling_distance)
    query_sequence = NCLTSequence(dataset_root, query_sequence, split=split, sampling_distance=sampling_distance)

    map_set = get_scans(map_sequence)
    query_set = get_scans(query_sequence)
    
    if sph:
        for i in tqdm.tqdm(range(len(query_set))):
            reading_filepath = query_set[i].rel_scan_filepath
            reading_filepath = os.path.join(dataset_root, reading_filepath)
            images = [load_im_file_for_generate(pc2image_file(reading_filepath, '/velodyne_sync/', i, '.bin')) for i in range(1, 6)]
            sph_filename = reading_filepath.replace('bin', 'jpg')
            sph_filename = sph_filename.replace('velodyne_sync', 'sph')
            sph_img = generate_sph_image(images, 'nclt', "/media/workspace/dataset/NCLT/image_meta.pkl", dataset_root)
            cv2.imwrite(sph_filename, sph_img)

        for i in tqdm.tqdm(range(len(map_set))):
            reading_filepath = map_set[i].rel_scan_filepath
            reading_filepath = os.path.join(dataset_root, reading_filepath)
            images = [load_im_file_for_generate(pc2image_file(reading_filepath, '/velodyne_sync/', i, '.bin')) for i in range(1, 6)]
            sph_filename = reading_filepath.replace('bin', 'jpg')
            sph_filename = sph_filename.replace('velodyne_sync', 'sph')
            sph_img = generate_sph_image(images, 'nclt', "/media/workspace/dataset/NCLT/image_meta.pkl", dataset_root)
            cv2.imwrite(sph_filename, sph_img)


    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    query_set = filter_query_elements(query_set, map_set, dist_threshold)

    print(f'{len(map_set)} database elements, {len(query_set)} query elements')
    return EvaluationSet(query_set, map_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for NCLT dataset')
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
    sequences = [('2012-02-04', '2012-03-17')]

    for map_sequence, query_sequence in sequences:
        print(f'Map sequence: {map_sequence}')
        print(f'Query sequence: {query_sequence}')

        test_set = generate_evaluation_set(args.dataset_root, map_sequence, query_sequence,
                                           sampling_distance=args.sampling_distance, dist_threshold=args.dist_threshold, sph=args.sph)

        pickle_name = f'test_{map_sequence}_{query_sequence}_{args.sampling_distance}.pickle'
        file_path_name = os.path.join(args.dataset_root, pickle_name)
        # test_set.save(file_path_name)
