"""Non-maximum suppression for edge detection results."""
import os
import cv2
import numpy as np
from scipy.io import loadmat
from ctypes import *

from .toolbox import conv_tri, grad2

_cxx_dir = os.path.join(os.path.dirname(__file__), '..', 'cxx', 'lib')
solver = cdll.LoadLibrary(os.path.join(_cxx_dir, 'solve_csa.so'))
c_float_pointer = POINTER(c_float)
solver.nms.argtypes = [c_float_pointer, c_float_pointer, c_float_pointer,
                       c_int, c_int, c_float, c_int, c_int]


def nms_process_one_image(image, save_path=None, save=True):
    """Run NMS on a single edge probability map."""
    if save and save_path is not None:
        assert os.path.splitext(save_path)[-1] == ".png"
    edge = conv_tri(image, 1)
    edge = np.float32(edge)
    ox, oy = grad2(conv_tri(edge, 4))
    oxx, _ = grad2(ox)
    oxy, oyy = grad2(oy)
    ori = np.mod(np.arctan(oyy * np.sign(-oxy) / (oxx + 1e-5)), np.pi)
    out = np.zeros_like(edge)
    r, s, m, w, h = 1, 5, float(1.01), int(out.shape[1]), int(out.shape[0])
    solver.nms(out.ctypes.data_as(c_float_pointer),
               edge.ctypes.data_as(c_float_pointer),
               ori.ctypes.data_as(c_float_pointer),
               r, s, m, w, h)
    edge = np.round(out * 255).astype(np.uint8)
    if save:
        cv2.imwrite(save_path, edge)
    return edge


def nms_process(result_dir, nms_dir, key=None, file_format=".mat"):
    """Run NMS on all results in a directory."""
    assert file_format in {".mat", ".npy"}
    assert os.path.isdir(result_dir)
    os.makedirs(nms_dir, exist_ok=True)
    for file in os.listdir(result_dir):
        save_name = os.path.join(nms_dir, "{}.png".format(os.path.splitext(file)[0]))
        if os.path.isfile(save_name):
            continue
        if os.path.splitext(file)[-1] != file_format:
            continue
        abs_path = os.path.join(result_dir, file)
        if file_format == ".mat":
            assert key is not None
            image = loadmat(abs_path)[key]
        elif file_format == ".npy":
            image = np.load(abs_path)
        else:
            raise NotImplementedError
        nms_process_one_image(image, save_name, True)
