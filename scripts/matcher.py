

import cv2
from matplotlib import pyplot as plt
from roma import roma_outdoor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

roma_model = roma_outdoor(device=device)
(imA_path, imB_path) = ('data/WLABS/ministry/01.png', 'data/WLABS/ministry/02.png')
img1 = cv2.imread(imA_path)
img2 = cv2.imread(imB_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

H_A, W_A = img1.shape[:2]
H_B, W_B = img2.shape[:2]

# Match
warp, certainty = roma_model.match(imA_path, imB_path, device=device)
# Sample matches for estimation
matches, certainty = roma_model.sample(warp, certainty)
# Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
# Find a fundamental matrix (or anything else of interest)
F, mask = cv2.findFundamentalMat(
    kptsA.cpu().numpy(), kptsB.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
)