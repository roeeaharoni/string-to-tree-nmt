import sys
import s2t_data


def main(input_path, output_path):
    s2t_data.bllip_parse(input_path, output_path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])