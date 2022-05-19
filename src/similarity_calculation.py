from sklearn import metrics as mr
from skimage.metrics import structural_similarity
import numpy as np
 
class MethodError(Exception):
    pass

def similarity_cal(img1, img2, method="mr"):
    if method == "mr":
        img1 = np.reshape(img1, -1)
        img2 = np.reshape(img2, -1)
        return mr.mutual_info_score(img1, img2)
    elif method == "ssim":
        return structural_similarity(img1, img2, multichannel=False)
    else:
        raise MethodError("mr or ssim")