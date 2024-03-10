import cv2 as cv
from knn_matting import Matting

if __name__ == '__main__':

    # load Matting
    mat = Matting()
    mat.setup()

    # 把 source, trimap, background 讀取進來
    source_image_path = 'images/gandalf.png'
    trimap_image_path = 'trimap/gandalf.png'
    background_image_path = 'landscape.jpg'
    source_image = cv.imread(source_image_path, 1)
    trimap = cv.imread(trimap_image_path, 1)
    background_image = cv.imread(background_image_path, 1)

    # 利用 knn matting 得到 alpha map
    alpha = mat.knn_matting(source_image, trimap)

    # 利用 alpha map 組合影像
    result_image = mat.combine_image(source_image, background_image, alpha)
    
    # 輸出結果
    cv.imshow("out", result_image)
    cv.waitKey(0)
    cv.imwrite("out.png", result_image)