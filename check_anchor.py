from utils.utils import kmean_anchors
file_list = 'input/dataset/fisheye/fisheye_9c_5016_train.txt'  # 5,11,  7,17,  16,13,  10,25,  15,37,  34,28,  25,62,  55,91,  114,93
kmean_anchors(file_list,n=9,img_size=(640,640), thr=0.35, gen=1000)

# When setting thr higher, can find more suitable anchor sizes..?
