import pickle
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--file', metavar='path', required=True,
                        help='path to file')

    args = parser.parse_args()


    with open(args.file, 'rb') as f:
        data = pickle.load(f)

    print(data)
