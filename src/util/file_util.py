# Copyright(c) Eric Steinberger 2018


import os
import os.path as osp


def get_all_files_in_dir(_dir):
    return [f for f in os.listdir(_dir) if osp.isfile(osp.join(_dir, f))]


def get_all_txt_files_in_dir(_dir):
    return [f for f in os.listdir(_dir) if osp.isfile(osp.join(_dir, f)) and f.endswith(".txt")]


def get_all_dirs_in_dir(_dir):
    return [d for d in os.listdir(_dir) if osp.isdir(osp.join(_dir, d))]


def create_dir_if_not_exist(path):
    if not osp.exists(path):
        os.makedirs(path)
