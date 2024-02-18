import pickle
import ee
import os
import retry
import requests
import shutil
import multiprocessing
from retry import retry
import logging
from src.utils import *
logging.basicConfig()


if __name__ == "__main__":
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    logging.basicConfig()


    points_lol = pickle.load(open("roi_samples_20240218-0150.pkl", 'rb'))
    list_of_points = []
    for region in points_lol:
        list_of_points += region

    print(f"num cores: {multiprocessing.cpu_count()}")


        



    pool = multiprocessing.Pool(25)

    pool.starmap(getResult, enumerate(list_of_points))
