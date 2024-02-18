import ee
import os
import multiprocessing
import pickle


ee.Authenticate()
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')



print("setting up")
dataset = ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019").select('discrete_classification')#ee.ImageCollection('ESA/WorldCover/v200').first()
ecoregions = ee.FeatureCollection("projects/data-sunlight-311713/assets/wwf_terr_ecos")


reduced_res = dataset.reduceResolution(ee.Reducer.mode(), True, 41).reproject(dataset.projection().atScale(640))

latam_filter = ee.Filter.Or(ee.Filter.eq("wld_rgn", "South America"), ee.Filter.eq("wld_rgn", "Central America"), ee.Filter.eq("country_na", "Mexico"))
latin_america = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(latam_filter)
#l2_areas = ee.FeatureCollection("FAO/GAUL/2015/level2").filterBounds(latin_america)
s2SamplingGrid = ee.FeatureCollection('projects/data-sunlight-311713/assets/sentinel_2_index_shapefile').map(lambda x: x.buffer(-320))

roi = ecoregions.filterBounds(latin_america)
print("getting ecoregion names")
ecoRegList = list(set(list(roi.aggregate_array("ECO_NAME").getInfo())))
print(ecoRegList    )
reduced_res = reduced_res.clip(roi)
print("got roi")

def sample_landcover(feat, tilescale=1):
    sample_roi = reduced_res.clip(feat.geometry())
    sample_geom = sample_roi.stratifiedSample(200, seed=1, region=feat, geometries=True, tileScale=tilescale)
    return sample_geom
print("generating samples...")

reqs_list = []
failed = []
for idx, name in enumerate(ecoRegList):
    try:
        print(f"starting {name}")
        _r = roi.filterMetadata("ECO_NAME", "equals", name).map(lambda x: x.buffer(100)).geometry()
        sample =    s2SamplingGrid.filterBounds(_r).map(lambda x: x.intersection(_r))
        features = sample_landcover(sample)
        geom = features.aggregate_array('.geo').getInfo()
        landcover = features.aggregate_array('discrete_classification').getInfo()
        reqs_list.append((geom, landcover))
        print(f"done with {idx+1} of {len(ecoRegList)}")
    except Exception as error:
        print(f"failed with {name}, {error}, trying again")
        try:
            _r = roi.filterMetadata("ECO_NAME", "equals", name).map(lambda x: x.buffer(100)).geometry()
            sample =    s2SamplingGrid.filterBounds(_r).map(lambda x: x.intersection(_r))
            features = sample_landcover(sample)
            geom = features.aggregate_array('.geo').getInfo()
            landcover = features.aggregate_array('discrete_classification').getInfo()
            reqs_list.append((geom, landcover))
            print(f"done with {idx+1} of {len(ecoRegList)}")
        except Exception as error2:
            failed.append(name)
            print(f"failed with {name}, {error}")
            continue

        

print("done with all regions")
print("done, saving...")
import datetime
pickle.dump(reqs_list, open(f'roi_samples_{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.pkl', 'wb'))
pickle.dump(failed,  open(f'failed_areas_{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.pkl', 'wb'))
print("done!")
