import ee
import os
import multiprocessing
import pickle


ee.Authenticate()
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')



print("setting up")
dataset = ee.ImageCollection('ESA/WorldCover/v200').first()
ecoregions = ee.FeatureCollection("projects/data-sunlight-311713/assets/wwf_terr_ecos")


reduced_res = dataset.reduceResolution(ee.Reducer.mode(), True, 4096).reproject(dataset.projection().atScale(640))

latam_filter = ee.Filter.Or(ee.Filter.eq("wld_rgn", "South America"), ee.Filter.eq("wld_rgn", "Central America"), ee.Filter.eq("country_na", "Mexico"))
latin_america = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(latam_filter)
l2_areas = ee.FeatureCollection("FAO/GAUL/2015/level2").filterBounds(latin_america)

roi = ecoregions.filterBounds(latin_america)

reduced_res = reduced_res.clip(roi)
print("got roi")

print("converting to list")
num_regions = roi.size().getInfo()
roi_list = ee.List(roi.toList(num_regions))

def sample_landcover(feat):
    sample_roi = reduced_res.clip(feat.geometry())
    sample_geom = sample_roi.stratifiedSample(10, seed=1, geometries=True)
    return sample_geom
print("generating samples...")
#reqs = roi.map(sample_landcover).flatten()
print("done")
print("getting geom")
reqs_list = []
for i in range(num_regions):
    reqs_list.append(sample_landcover(ee.Feature(roi_list.get(i))).aggregate_array('.geo').getInfo())
    print(f"done with {i+1} of {num_regions}")

#reqs=sample_landcover(roi.first()) # for testing

#reqs = reqs.getInfo()
print("done, saving...")
pickle.dump(reqs_list, open('roi_samples_list.pkl', 'wb'))
print("done!")
#print(reqs)