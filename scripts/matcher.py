

import IPython
import cv2
from matplotlib import pyplot as plt
import numpy as np
from roma import roma_outdoor
import torch
import codecs, json

from kornia_moons.viz import *
import kornia.feature as KF

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

def save_roma(file_path, F, kps1, kps2, inliers):
    FM = {
      "F": F.tolist(),
      "kps1": kps1.tolist(),
      "kps2": kps2.tolist(),
      "inliers": inliers.tolist()
    }

    json.dump(FM, codecs.open(file_path, 'w', encoding='utf-8'),
            separators=(',', ':'),
            sort_keys=False,
            indent=4)

def load_roma(file_path):
    transform_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    FM_new = json.loads(transform_text)
    F = np.array(FM_new["F"])

    kps1 = np.array(FM_new["kps1"])
    kps2 = np.array(FM_new["kps2"])
    inliers = np.array(FM_new["inliers"])
    return F, kps1,kps2, inliers

####################################################################################
file_path = "transform.json"
device = torch.device("cuda")

roma_model = roma_outdoor(device=device)
(imA_path, imB_path) = ('data/antene_cross.bmp', 'data/antene_cross_google.png')
(imA_path, imB_path) = ('data/WxBS/WLABS/ministry/01.png', 'data/WxBS/WLABS/ministry/02.png')

img1 = cv2.imread(imA_path)
img2 = cv2.imread(imB_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

if True:
  with torch.inference_mode():
      F, kps1, kps2, inliers = estimate_F_ROMA(img1, img2)

  save_roma(file_path, F, kps1, kps2, inliers)
else:
  F, kps1, kps2, inliers = load_roma(file_path)

fig, axes = plt.subplots(1, 3, figsize=(10, 10))

draw_LAF_matches(
      KF.laf_from_center_scale_ori(torch.from_numpy(kps1[None]).cpu()),
      KF.laf_from_center_scale_ori(torch.from_numpy(kps2[None]).cpu()),
      np.concatenate([np.arange(len(kps1)).reshape(-1,1), np.arange(len(kps2)).reshape(-1,1) ], axis=1),
      img1, img2,
      inliers.astype(bool),
      draw_dict={"inlier_color": (0.2, 1, 0.2, 0.2), "tentative_color": ( 1, 0.2, 0.3, 0.2), "feature_color": None, "vertical": True},
      return_fig_ax=False,
      ax=axes[0], fig=fig
      )

# put marker on image and transfer it to another image
width, height = img1.shape[:2]
center_x = height // 2
center_y = width // 2

# Draw a red dot in the center
source_image = cv2.circle(img1.copy(), (center_x, center_y), 50, (0, 0, 255), -1)  # Red color

red_dot = np.array([[center_x, center_y, 1]], dtype=np.float32)
blue_dot = F @ red_dot.T
blue_dot /= blue_dot[2]

# Draw the transferred red dot on the target image
transferred_image = cv2.circle(img2.copy(), (int(blue_dot[0]), int(blue_dot[1])), 15, (0, 0, 255), -1)

axes[1].imshow(source_image)
axes[2].imshow(transferred_image)
plt.show()
