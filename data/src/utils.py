import ee
import os
import retry
import requests
import shutil
from retry import retry
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
import logging
logging.basicConfig()


def min_cloudy_img(input):
    mask = ee.Image.constant(input.get("CLOUDY_PIXEL_PERCENTAGE"))
    clouds = ee.Image.constant(1).subtract(mask).cast({"constant":"float"})
    return input.addBands([clouds])



@retry(tries=10, delay=1, backoff=2)
def getResult(index, point):
    """Handle the HTTP requests to download an image."""
        
    #  point = point['coordinates']
    # Generate the desired image from the given point.
    point = ee.Geometry.Point(point['coordinates'])
    region = point.buffer(310).bounds()
    s2 = ee.ImageCollection("COPERNICUS/S2_SR").filterDate("2019-01-01", "2020-01-01")
    s2_mincloud_band = s2.map(min_cloudy_img)
    noCloudMosaic = s2_mincloud_band.qualityMosaic("constant").clip(region).select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A' ,'B9', 'B11', 'B12'])
    defaultProj = noCloudMosaic.projection()

    image = noCloudMosaic.clip(region)

    # Fetch the URL from which to download the image.
    url = image.getDownloadURL({
        'region': region,
        'dimensions':'64x64',
        #'scale':10,
        'format': 'GEO_TIFF'})

    # Handle downloading the actual pixels.
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()

    DestDir = f"DownloadedDataset/"

    if not os.path.exists(DestDir):
        os.mkdir(DestDir)

    filename = DestDir+f'tile_{index}.tif'
    with open(filename, 'wb') as out_file:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, out_file)
    print("Done: ", index)
