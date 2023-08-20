import pytest
import spotFinder
import setupLoaders
import dataLoader
import condition
import numpy as np
import ast

def func(x):
    return x + 1


def test_answer():
    assert func(3) == 4

def test_prepro_works():
    #make a numpy array with pilatus dimensions
    img = np.random.randint(1001, size=(2527, 2463))

    with open('condition.py', 'r') as f:
        for cond_meth in [node for node in ast.walk(f) if isinstance(node, ast.FunctionDef)]:
            new_img = setupLoaders.processImage(img=img, cond_meth=cond_meth) #pre-process image
    assert True
    #preprocess it the same way as setup logger
    #verify it is not just 0

