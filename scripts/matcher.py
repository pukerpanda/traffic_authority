

import cv2
from matplotlib import pyplot as plt
from roma import roma_outdoor
import torch

device = torch.device("cuda")

roma_model = roma_outdoor(device=device)
(imA_path, imB_path) = ('data/WxBS/WLABS/ministry/01.png', 'data/WxBS/WLABS/ministry/02.png')
img1 = cv2.imread(imA_path)
img2 = cv2.imread(imB_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# H_A, W_A = img1.shape[:2]
# H_B, W_B = img2.shape[:2]

# # Match
# warp, certainty = roma_model.match(imA_path, imB_path, device=device)
# # Sample matches for estimation
# matches, certainty = roma_model.sample(warp, certainty)
# # Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
# kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
# # Find a fundamental matrix (or anything else of interest)
# F, mask = cv2.findFundamentalMat(
#     kptsA.cpu().numpy(), kptsB.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
# )

def estimate_F_ROMA(img1, img2):
    # device=torch.device('cuda')
    #disk = KF.DISK.from_pretrained("depth").to(device)
    H_A, W_A = img1.shape[:2]
    H_B, W_B = img2.shape[:2]

    cv2.imwrite("img1.png", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    cv2.imwrite("img2.png", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    warp, certainty = roma_model.match("img1.png", "img2.png", device=device)
    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty, num=2000)
    # Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
    kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    kps1 = kptsA.detach().cpu().numpy()
    kps2 = kptsB.detach().cpu().numpy()
    try:
      F, inliers = cv2.findFundamentalMat(kps1, kps2, cv2.USAC_MAGSAC, 0.25, 0.999, 100000)
    except:
      F = np.array([[0.0, 0.0, 0.0],
              [0.0, 0.0, -1.0],
              [0.0, 1.0, 0.0]])
    if F is None:
      F = np.array([[0.0, 0.0, 0.0],
              [0.0, 0.0, -1.0],
              [0.0, 1.0, 0.0]])
    return F, kps1 ,kps2, inliers

with torch.inference_mode():
    F, kps1, kps2, inliers = estimate_F_ROMA(img1, img2)

IPython.embed()