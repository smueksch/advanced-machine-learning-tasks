import matplotlib.pyplot as plt


def visualize_segmentation(ax, image, segmentation, segmentation_opacity=0.5):
    ax.imshow(image)
    ax.imshow(segmentation, alpha=segmentation_opacity)
