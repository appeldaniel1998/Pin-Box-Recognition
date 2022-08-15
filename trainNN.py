import torch
import torchvision
from roboflow import Roboflow


rf = Roboflow(api_key="cNpdg5tikNEVov7CYRVN")
project = rf.workspace("ariel-ha3vz").project("pin-box")
dataset = project.version(2).download("yolov5")

