from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from mlfromscratch.model.clustering import clustering
import time

# set up the parameters prior to running the program
params = {
    "k": 2,
    "method": 'kmeans',  # kmeans or kmedoids
    "file_path": "data/taipei.jpg",
}


def read_img(path):
    """
    Read image and store it as an array, given the image path.
    :return: the 3 dimensional image array.
    """
    img = Image.open(path)
    img_arr = np.array(img, dtype='int32')
    img.close()
    return img_arr


def display_image(arr, k):
    """
    Display the image
    """
    arr = arr.astype(dtype='uint8')
    img = Image.fromarray(arr, 'RGB')
    plt.imshow(np.asarray(img))
    ax = plt.gca()
    ax.xaxis.set_ticks(np.arange(0, 200, 100))
    ax.yaxis.set_ticks(np.arange(0, 150, 50))
    ax.tick_params(axis='both', which='major', labelsize=18)
    title = "k = {}".format(k)
    ax.set_title(title, fontsize=20)
    plt.show()


def plot_log(x, y):
    """
    Create the plot of y versus x
    """
    plt.figure(figsize=(8, 8))
    plt.plot(x, y)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('WCSS', fontsize=18)
    plt.show()


if __name__ == '__main__':
    # read the image
    img_arr = read_img(params['file_path'])
    img_reshaped = img_arr.reshape(-1, 3)

    # set up the parameters for clustering
    k = params['k']
    method = params['method']
    print(
        "size of the testing set: {}, dimension: {}, k: {}, method: {}".format(img_arr.size, img_arr.shape, k, method))

    # kick up the clustering
    start_time = time.time()
    labels, centers, log = clustering(img_reshaped, k, starting_centers=None, method=method)
    print("--- %s seconds ---" % (time.time() - start_time))

    # visualize the result
    x, y = [i[0] for i in log], [i[1] for i in log]
    plot_log(x, y)
    img_clustered = np.array([centers[i] for i in labels])
    r, c, l = img_arr.shape
    img_disp = np.reshape(img_clustered, (r, c, l), order="C")
    display_image(img_disp, k)
