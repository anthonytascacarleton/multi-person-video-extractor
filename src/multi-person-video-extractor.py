import os
import random
import argparse
from shutil import copy
from extractors import VideoFeatureExtractor

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--input_folder",
            help="Location of the foler with videos")
    parser.add_argument(
            "--output_folder",
            help="The folder to output the multi-person videos",
            default="./processing/")
    parser.add_argument(
            "--num_sample",
            help="The number of videos to randomly sample from input_folder",
            type=int,
            default=1)
    parser.add_argument(
            "--use_gpu",
            help="Use a GPU for extraction",
            dest='use_gpu',
            default=False,
            action='store_true')
    parser.add_argument(
            "--display_results",
            help="Display facial detection results while extracting videos",
            dest='display_results',
            default=False,
            action='store_true')
    parser.add_argument(
            "--single_person_mode",
            help="Extract single-person videos instead of multi-person videos",
            dest='single_person_mode',
            default=False,
            action='store_true')
    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists(args.input_folder):
        print("Input folder not found: {0}".format(args.input_folder))
        exit()
    if not os.path.exists(args.output_folder):
        print("Output folder not found: {0}".format(args.output_folder))
        exit()
    extracted_video_counter = 0
    video_feature_extractor = VideoFeatureExtractor(use_gpu=args.use_gpu)
    filenames=list()
    for filename in [args.input_folder + f for f in os.listdir(args.input_folder) if os.path.isfile(os.path.join(args.input_folder, f))]:
        filenames.append(filename)
    print("{0} files found in {1}".format(len(filenames), args.input_folder))
    random.shuffle(filenames)
    for filename in filenames:
        if not filename.endswith('.mp4'):
            continue
        is_multi_person_video = video_feature_extractor.is_multi_person_video(video_file=filename, display_results=args.display_results)
        if (is_multi_person_video and not args.single_person_mode) or (not is_multi_person_video and args.single_person_mode):
            copy(filename, args.output_folder)
            extracted_video_counter = extracted_video_counter + 1
            print('number of extracted videos = {0}'.format(extracted_video_counter))
        if (extracted_video_counter == args.num_sample):
            break