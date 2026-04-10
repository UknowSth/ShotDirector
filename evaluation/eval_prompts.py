import json

with open('evaluation/prompts.json', 'r') as fcc_file:
    prompts = json.load(fcc_file)

def adjust_intrinsics(fx, fy, cx, cy, ori_h, ori_w, tar_h=480, tar_w=832):
    # Step 1: scale factor
    scale = max(tar_h / ori_h, tar_w / ori_w)
    H_resized = ori_h * scale
    W_resized = ori_w * scale

    fx *= scale
    fy *= scale
    cx *= scale
    cy *= scale

    # Step 2: center crop offset
    i = (H_resized - tar_h) / 2
    j = (W_resized - tar_w) / 2

    # cx -= j
    # cy -= i
    # cx = tar_h  / 2
    # cy = tar_w / 2
    cx = tar_w  / 2
    cy = tar_h / 2

    return fx, fy, cx, cy

for key, prompt in prompts.items():
    if 'width' in prompt:
        fx, fy, cx, cy = adjust_intrinsics(prompt['intrinsic'][0][0][0], prompt['intrinsic'][0][1][1], prompt['intrinsic'][0][0][2], prompt['intrinsic'][0][1][2], prompt['height'], prompt['width'], 480, 832)
        prompt['intrinsic'][0][0][0] = fx
        prompt['intrinsic'][0][1][1] = fy
        fx, fy, cx, cy = adjust_intrinsics(prompt['intrinsic'][1][0][0], prompt['intrinsic'][1][1][1], prompt['intrinsic'][1][0][2], prompt['intrinsic'][1][1][2], prompt['height'], prompt['width'], 480, 832)
        prompt['intrinsic'][1][0][0] = fx
        prompt['intrinsic'][1][1][1] = fy