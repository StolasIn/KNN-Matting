from knn_matting import Matting
import os
import cv2 as cv
import timeit, functools

image_folder = 'images'
trimap_folder = 'trimap'
background = cv.imread('landscape.jpg', 1)

def get_images(path):
    file_names = []
    images = []
    for file_name in os.listdir(path):
        file_names.append(file_name)
    
    file_names = sorted(file_names)
    for file_name in file_names:
        image_path = os.path.join(path, file_name)
        image = cv.imread(image_path, 1)
        images.append(image)
    
    return file_names, images

def experiment_k(image_names, images, trimaps, ks = []):
    mat = Matting()
    for k in ks:
        for i in range(len(images)):
            mat.setup(n_neighbors=k)
            alpha = mat.knn_matting(images[i], trimaps[i])
            result_image = mat.combine_image(images[i], background, alpha)
            cv.imwrite(f"experiment_different_k/result/{image_names[i]}_{k}.png", result_image)
            cv.imwrite(f"experiment_different_k/alpha/{image_names[i]}_{k}_alpha.png", alpha*255)

def experiment_encode(image_names, images, trimaps):
    mat = Matting()
    dtypes = ['RGBxy_rand', 'RGBxy', 'RGB', 'HSVxy', 'HSV']
    for dtype in dtypes:
        for i in range(len(images)):
            mat.setup(feature_representation = dtype)
            alpha = mat.knn_matting(images[i], trimaps[i])
            result_image = mat.combine_image(images[i], background, alpha)
            cv.imwrite(f"experiment_different_encode/result/{image_names[i]}_{dtype}.png", result_image)
            cv.imwrite(f"experiment_different_encode/alpha/{image_names[i]}_{dtype}_alpha.png", alpha*255)

def experiment_lambda(image_names, images, trimaps, lambdas = []):
    mat = Matting()
    for l in lambdas:
        for i in range(len(images)):
            mat.setup(_lambda = l)
            alpha = mat.knn_matting(images[i], trimaps[i])
            result_image = mat.combine_image(images[i], background, alpha)
            cv.imwrite(f"experiment_different_lambda/result/{image_names[i]}_{l}.png", result_image)
            cv.imwrite(f"experiment_different_lambda/alpha/{image_names[i]}_{l}_alpha.png", alpha*255)

def _experiment_slow(image, trimap):
    mat = Matting()
    mat.setup(use_umfpack=False)
    _ = mat.knn_matting(image, trimap)

def _experiment_speedup(image, trimap):
    mat = Matting()
    mat.setup(use_umfpack=True)
    _ = mat.knn_matting(image, trimap)

def experiment_speed(times):
    small_image = cv.imread('speedup/small.png', 1)
    small_trimap = cv.imread('speedup/trimap_small.png', 1)
    large_image = cv.imread('speedup/large.png', 1)
    large_trimap = cv.imread('speedup/trimap_large.png', 1)
    small_slow = timeit.Timer(functools.partial(_experiment_slow, small_image, small_trimap))
    small_fast = timeit.Timer(functools.partial(_experiment_speedup, small_image, small_trimap))
    large_slow = timeit.Timer(functools.partial(_experiment_slow, large_image, large_trimap))
    large_fast = timeit.Timer(functools.partial(_experiment_speedup, large_image, large_trimap))

    print(f'small image:\nw/o LU factorization : {small_slow.timeit(times):.2f} seconds\nw/ LU factorization : {small_fast.timeit(times):.2f} seconds\n')
    print(f'large image:\nw/o LU factorization : {large_slow.timeit(times):.2f} seconds\nw/ LU factorization : {large_fast.timeit(times):.2f} seconds\ntest time : {times}')

def work_more_images():
    image_path = 'dataset/input_lowres'
    trimap_path = 'dataset/trimap_lowres/Trimap1'

    names, images = get_images(image_path)
    _, trimaps = get_images(trimap_path)

    mat = Matting()
    mat.setup(n_neighbors=30)
    for i in range(len(images)):
        alpha = mat.knn_matting(images[i], trimaps[i])
        result_image = mat.combine_image(images[i], background, alpha)
        cv.imwrite(f"work_images/result/{names[i]}.png", result_image)
        cv.imwrite(f"work_images/alpha/{names[i]}_alpha.png", alpha*255)

def test_sci(names, images, trimaps):
    mat = Matting()
    mat.setup()
    for i in range(len(images)):    
        alpha = mat.knn_matting(images[i], trimaps[i])
        result_image = mat.combine_image(images[i], background, alpha)
        cv.imwrite(f"sci_trimap/result/{names[i]}.png", result_image)
        cv.imwrite(f"sci_trimap/alpha/{names[i]}_alpha.png", alpha*255)


if __name__ == '__main__':
    names, images = get_images(image_folder)
    _, trimaps = get_images(trimap_folder)
    ks = [1, 5, 10, 25, 100]
    experiment_k(names, images, trimaps, ks)