import numpy as np
import vis_tools
from pathlib import Path as P
from glob import glob

import typing
from typing_extensions import Literal, TypedDict

# Force these into the standard typing module for older Python versions
typing.Literal = Literal
typing.TypedDict = TypedDict

from vod.visualization.settings import label_color_palette_2d

if __name__ == '__main__':
    path_dict = {
        'CFAR_radar':'results',
        'radar_rcsv':'output/IA-SSD-vod-radar/iassd_best_aug_new/eval/best_epoch_checkpoint',
        'radar_rcs':'output/IA-SSD-vod-radar/iassd_rcs/eval/best_epoch_checkpoint',
        'radar_v':'output/IA-SSD-vod-radar/iassd_vcomp_only/eval/best_epoch_checkpoint',
        'radar':'output/IA-SSD-vod-radar-block-feature/only_xyz/eval/best_epoch_checkpoint',
        'lidar_i':'output/IA-SSD-vod-lidar/all_cls/eval/checkpoint_epoch_80',
        'lidar':'output/IA-SSD-vod-lidar-block-feature/only_xyz/eval/best_epoch_checkpoint',
        'CFAR_lidar_rcsv':'output/IA-SSD-GAN-vod-aug-lidar/to_lidar_5_feat/eval/best_epoch_checkpoint',
        'CFAR_lidar_rcs':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_rcs_only/eval/best_epoch_checkpoint',
        'CFAR_lidar_v':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_vcomp_only/eval/best_epoch_checkpoint',
        'CFAR_lidar':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_xyz_only/eval/best_epoch_checkpoint'
    }

    lidar_range = [0, -25.6, -3, 51.2, 25.6, 2]
    abs_path = P(__file__).parent.resolve()
    base_path = abs_path.parents[1]

    tag = 'CFAR_radar'

    # --- UPDATED PATHS ---
    # We define the master data root exactly where you specified
    DATA_ROOT = P('/root/data/view_of_delft_PUBLIC/radar/')

    vis_root_path = base_path / path_dict[tag]
    test_result_path = vis_root_path / 'final_result' / 'data'
    test_result_files = sorted(glob(str(test_result_path / '*.txt')))
    
    is_radar = False if 'lidar' in tag.lower() else True
    is_test = True
    split = 'testing' if is_test else 'training'
    
    # Force the pcd path to look inside your requested radar directory
    pcd_file_path = DATA_ROOT / split / 'velodyne'

    def get_frame_id(fname):
        fname_p = P(fname)
        frame_id = str(fname_p.name).split('.')[0]
        return frame_id

    def collect_result_dict(result_files):
        result_dict = {}
        for fname in result_files:
            frame_id = get_frame_id(fname)
            annas = vis_tools.load_result_file(fname)
            result_dict[frame_id] = annas
        return result_dict
    
    annas_dict = collect_result_dict(test_result_files)
    print(annas_dict['08466'])

    gt_annos_dict = None
    if not is_test:
        gt_label_path = DATA_ROOT / split / 'label_2'
        gt_files = sorted(glob(str(gt_label_path / '*.txt')))
        gt_annos_dict = collect_result_dict(gt_files)
        print(f"DEBUG: Found {len(gt_files)} Ground Truth label files.")

    frame_ids = sorted(list(annas_dict.keys()))

    cls_name = ['Car','Pedestrian', 'Cyclist']
    color_dict = {}
    for i, v in enumerate(cls_name):
        # Applied the .get() fallback to prevent KeyErrors for missing classes
        color_dict[v] = label_color_palette_2d.get(v, (128, 128, 128))

    # Output paths routed to the new radar directory
    vis_img_path = DATA_ROOT / split / 'output_images'
    vis_img_path.mkdir(exist_ok=True, parents=True)

    img_title = 'lidar' if 'lidar' in tag else 'radar'
    
    # Route video saves to the new directory
    vid_path = DATA_ROOT / 'output_videos'
    vid_path.mkdir(exist_ok=True, parents=True)
    vid_fname = vid_path / (f'{tag}.mp4')
    
    if not vid_fname.exists():
        # print('saving detection BEV video...')
        # print(f"DEBUG: Attempting to save BEV video to {vid_fname}")
        # print(f"DEBUG: Found {len(test_result_files)} detection result files to visualize.")
        # print(f"DEBUG: PCD file: {pcd_file_path}") 
        # print(f"DEBUG: Visualization output path: {vis_img_path}")
        vis_tools.saveODImgs(frame_ids, annas_dict, pcd_file_path, vis_img_path, color_dict,\
            is_radar, title=img_title, limit_range=lidar_range, is_test=is_test)
        
        dt_imgs = sorted(glob(str(vis_img_path/'*.png')))
        
        # Added safety check to prevent 'NoneType has no attribute release' crash
        if len(dt_imgs) > 0:
            vis_tools.make_vid(dt_imgs, vid_fname, fps=10)
        else:
            print("Warning: No BEV images found to compile into a video!")

    # save rgb
    rgb_vid = vid_path / 'rgb.mp4'
    if rgb_vid.exists():
        pass
    else:
        print('saving rgb video...')
        img_files = [vis_tools.get_img_file(id, is_radar=is_radar, is_test=is_test) for id in frame_ids]
        
        # Filter out any None paths in case vis_tools couldn't find the RGB images
        img_files = [img for img in img_files if img is not None]
        
        if len(img_files) > 0:
            vis_tools.make_vid(img_files, rgb_vid, fps=10)
        else:
            print("Warning: No RGB images found to compile into a video!")