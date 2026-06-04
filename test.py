from scipy.io import loadmat

from dataLoader import MIT1003Dataset

dataset = MIT1003Dataset(
    "dataset/MIT1003"
)

image, heatmap, fixation = dataset[0]

print(image.shape)
print(heatmap.shape)
print(fixation.shape)

print(fixation.unique())