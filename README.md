# KNN_Matting

![](/matting.png)

## usage

modify image path in main.py then run `python main.py`

```
    source_image_path = 'images/gandalf.png'
    trimap_image_path = 'trimap/gandalf.png'
    background_image_path = 'landscape.jpg'
```

## hyper-parameters

- feature_representation: how to represent a pixel
- n_neighbors: number of neighbors of KNN
- _lambda: certainty of given foreground pixels
- knn_type: sklearn or hand_craft
- metrix_type: sparse or dense
- use_umfpack: speed up of spsolve function in scipy