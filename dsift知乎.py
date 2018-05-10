import  numpy as np
from  PIL import  Image
import pylab as pl
import dsift
import sift


dsift.process_image_dsift_2('../data/empire.jpg','empire.sift',90,40,True)
l,d = sift.read_features_from_file('empire.sift')

im = np.array(Image.open('../data/empire.jpg'))

sift.plot_features(im,l,True)

pl.show()
