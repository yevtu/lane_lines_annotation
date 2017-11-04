from annotator import Annotator
import argparse

def main():
    parser = argparse.ArgumentParser(description="Lane Lines annotation")
    parser.add_argument('-i',
                        '--in-path',
                        help='Input video path')
    parser.add_argument('-o',
                        '--out-path',
                        help='Path for annotated video (including its name)')
    args = parser.parse_args()

    ann = Annotator(args.in_path, args.out_path)
    ann.process_video()

    return

if __name__ == "__main__":
    main()