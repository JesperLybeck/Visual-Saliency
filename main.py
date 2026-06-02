from dataset import SaliconDataset

dataset = SaliconDataset(
    split="train",
    transform=None
)

image, heatmap, fixation_map = dataset[0]

print("\nShapes")
print("Image:", image.shape)
print("Heatmap:", heatmap.shape)
print("Fixation:", fixation_map.shape)

print("\nRanges")
print("Image min/max:", image.min().item(), image.max().item())
print("Heatmap min/max:", heatmap.min().item(), heatmap.max().item())

print("\nFixation statistics")
print("Unique values:", fixation_map.unique())
print("Num fixation pixels:", fixation_map.sum().item())