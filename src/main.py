import argparse
from unsupervised_tracker import UnsupervisedTracker

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", help="Path to video in which an object must be tracked")
parser.add_argument("--train",
                    help="Train the network required to recognize objects in video",
                    action="store_true")
args = parser.parse_args()

if not (args.video or args.train):
    parser.error('No action requested, add --video or --train')
else:
    unsupervised_tracker = UnsupervisedTracker()
    unsupervised_tracker.track_object_in_video(args.video, args.train)
