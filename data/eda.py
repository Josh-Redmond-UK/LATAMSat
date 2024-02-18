import pickle

roi = pickle.load(open("roi_samples_20240218-0150.pkl", 'rb'))

print(roi)
print(len(roi))
print(roi[0])
print(len(roi[0]))

list_of_points = []
for region in roi:
    list_of_points += region


print(list_of_points)
print(len(list_of_points))