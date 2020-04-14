import os
import cv2
import pickle

def read(file_path):
    with open(file_path, 'rb') as f:
        output = pickle.load(f)
        image_source = output['img_ori']
        image_distorted = output['img']
        grid_gt = output['grid_back']
    return image_source, image_distorted, grid_gt

if __name__ == '__main__':
    file_root = './DOC_SYN'
    file_save = './DOC_SYN_save'
    if not os.path.isdir(file_save):
        os.makedirs(file_save)
    file_list = sorted(os.listdir(file_root))
    for i in range(len(file_list)):
        image_source, image_distorted, grid_gt = read(os.path.join(file_root, file_list[i]))
        print('i:', i, grid_gt)
        cv2.imwrite(file_save + '/%05d_source.png' % (i), image_source)
        cv2.imwrite(file_save + '/%05d_distorted.png' % (i), image_distorted)



