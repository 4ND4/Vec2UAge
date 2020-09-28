# Cropped face images to face vectors
import os
from facenet import get_vectors
from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, help="Path with cropped face images", required=True)
    parser.add_argument("--out_file", type=str, help="JSON mapping filename to face vector", required=True)
    args = parser.parse_args()

    num_processed = get_vectors(input_path=args.input, output_path=args.out_file, image_size=160)
    print(f"Processed {num_processed} images from {args.input}, see results in {args.outfile}")
    