# import time
import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2
from scipy.stats import itemfreq
from skimage.feature import local_binary_pattern
import scipy.cluster.vq as vq
from lbp import *
import pickle as pkl
import libsvm
import random

def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1))
    return histogram_of_words

if __name__ == '__main__':
    preprocess = True
    codebook = True
    extractFeature = True
    # hog_flag = True
    # orb_flag = False
    feature_flag = "hog"
    path = 'Base11/'
    new_path='Base11_resized/'

    trainLabels = pd.read_csv('Base11/Annotation_Base11_3.csv')
    labels = np.array(pd.read_csv('Base11/Annotation_Base11_3.csv', usecols = ['Retinopathy grade'])).reshape(-1)
    # exit()
    
    print("Loaded Base11")

    dirs = [l for l in os.listdir(path) if l != '.DS_Store']

    if preprocess is True:
        cropx=1400
        cropy=1400    

        if not os.path.exists(new_path):
            os.makedirs(new_path)



        for item in dirs:
            name = item.split(".")
            if (name[1] == "tif"):
                # img = io.imread(path+item)
                img = cv2.imread(path+item)
                
                #crop images
                # startx = x//2-(cropx//2) - 10
                # starty = y//2-(cropy//2) + 10
                x_start_temp = np.max(img[:,:,2], axis = 1)
                x_start_index = np.where(x_start_temp > 40)[0][0]
                x_end_index = np.where(x_start_temp > 40)[0][-1]

                y_start_temp = np.max(img[:,:,2], axis = 0)
                y_start_index = np.where(y_start_temp > 40)[0][0]
                y_end_index = np.where(y_start_temp > 40)[0][-1]

                img = img[x_start_index:x_end_index, y_start_index:y_end_index, :]
                
                #resize image to 256x256
                img = cv2.resize(img, (256,256))
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))
                r, g, b = cv2.split(img)
                # cl1 = clahe.apply(r)
                cl2 = clahe.apply(g)
                img = cl2
                # cl3 = clahe.apply(b)
                # img =cv2.merge((cl1, cl2, cl3))
                img = cl2 - cv2.medianBlur(cl2, 5)
                cv2.imwrite(f"{new_path}{item}",img)
                # break

        # print("Resized Images")
    print("preprocessing done")

    lst_imgs = [l for l in trainLabels['Image name']]
    # c = list(zip(lst_imgs, labels))
    # random.shuffle(c)
    # lst_imgs, labels = zip(*c)

    # cutoff = int(len(lst_imgs)*0.4)
    # # print(lst_imgs)
    # lst_imgs= lst_imgs[0:cutoff]
    # labels = labels[0:cutoff]
    X = np.array([np.array(Image.open(new_path + img)) for img in lst_imgs])
    # print(labels[0])
    features_dict = []
    allfeatures = []
    if codebook is True:
        hog = cv2.HOGDescriptor("hog.xml")
        orb = cv2.ORB_create(400)
        for i in range(X.shape[0]):
            if feature_flag is "hog":
                feature = hog.compute(X[i]).reshape(-1,31)
                features_dict.append(feature)
                allfeatures.extend(feature)
            elif feature_flag is "orb":
                keyPoints, desc = orb.detectAndCompute(X[i], None)
                features_dict.append(desc)
                allfeatures.extend(desc)
        
        #allfeature is just features_dict but just it is merged across first dimension        
        allfeatures = np.array(allfeatures).astype(np.float64)
        # allfeatures = allfeatures.reshape(allfeatures.shape[0]*allfeatures.shape[1], allfeatures.shape[2])
        
        num_features = allfeatures.shape[0]
        # print(num_features)
        num_clusters = 100#int(num_features**0.5)

        #codebook contains information about clusters
        codebook, _ = vq.kmeans(allfeatures, num_clusters, thresh = 1)
        with open(f"codebook_{feature_flag}", "wb") as f:
            d = {}
            d["codebook"] = codebook
            d["features_dict"] = features_dict
            pkl.dump(d, f)
    else:
        with open(f"codebook_{feature_flag}", "rb") as f:
            d = pkl.load(f)
            codebook = d["codebook"]
            features_dict = d["features_dict"]
    # print(features_dict.shape)
    print("codebook done")

    print(len(lst_imgs), len(labels), len(X), len(features_dict))
    if extractFeature is True:
        histogram_per_image = []

        for i in range(X.shape[0]):
            histogram_per_image.append(computeHistograms(codebook, features_dict[i]))
            
        histogram_per_image = np.array(histogram_per_image)
        with open(f"features_{feature_flag}.pk", "wb") as f:
            d = {}
            d["data"] = histogram_per_image
            d["labels"] = labels
            pkl.dump(d, f)

    #different features extractors
    # else:
    #     with open("features_hog.pkl", 'rb') as f:
    #         d = pkl.load(f)
    # c, g, rate, model_file = libsvm.grid(datasetpath + HISTOGRAMS_FILE, png_filename='grid_res_img_file.png')

    # hogfeature = hog.compute(X[0])
    # hogfeature = hogfeature.reshape(-1,31)
    # print(hogfeature.shape)

    # surf = cv2.ORB_create(400)#cv2.SURF(400)

    # # surf.extended = True

    # keyPoints, desc = surf.detectAndCompute(X[0], None)
    # # print(keyPoints)
    # # print(len(keyPoints))
    # # print(desc)
    # # print(desc.shape)
    # lbp = local_binary_pattern(X[0], 56, 5, method='uniform')
    # x = itemfreq(lbp.ravel())
    # # Normalize the histogram
    # print(x.shape)