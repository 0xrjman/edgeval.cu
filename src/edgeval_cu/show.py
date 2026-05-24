"""Show evaluation results — display ODS/OIS rankings."""
import os
import argparse


def get_flist(dir_str, full):
    dirs_ = set()
    file_dirs = dir_str.split(' ')
    for file_dir in file_dirs:
        for root, dirs, files in os.walk(file_dir):
            if not full and root.split("/")[-1] == "nms-eval-9" and 'eval_bdry.txt' in files:
                dirs_.add(os.path.join(root, 'eval_bdry.txt'))
            elif full and root.split("/")[-1] == "nms-eval" and 'eval_bdry.txt' in files:
                dirs_.add(os.path.join(root, 'eval_bdry.txt'))
    return dirs_


def main(args):
    ODS_list = []
    OIS_list = []
    result_list = []
    for txt_path in get_flist(args.dir, args.full):
        with open(txt_path) as file:
            context = file.readline().split()
        ODS_list.append((txt_path, float(context[3])))
        OIS_list.append((txt_path, float(context[6])))
        result_list.append((txt_path, float(context[3]), float(context[6])))
    ODS_list.sort(key=lambda a: a[1], reverse=True)
    OIS_list.sort(key=lambda a: a[1], reverse=True)
    result_list.sort(key=lambda a: a[1], reverse=True)
    print("==" * 20)
    print("epoch          ODS            OIS")
    print("--" * 20)
    if len(result_list) == 0:
        print("\nThere is no avaliable result to show\n")
    else:
        for i in result_list:
            print("{:10}{:12}{:16}".format(i[0].split('/')[-3], i[1], i[2]))
    print("==" * 20 + "\n")
