import os
import time
import argparse
import warnings

# Suppress pydantic warnings from lerobot
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import numpy as np
import random
import datetime
import torch

import utils.utils as utils
from env.constants import WORKSPACE_LIMITS
from env.environment_sim import Environment
from helpers.logger import Logger
from action_generator.place_generator import Placenet
from feature_extractor.feature_field_builder import FeatureField
from helpers.dataset import CLIPActionDataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=234, metavar='N',
                    help='random seed (default: 234)')

    parser.add_argument('--evaluate', dest='evaluate', action='store_true', default=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
    parser.add_argument('--sample_grasp', dest='sample_grasp', action='store_true', default=False)
    parser.add_argument('--sample_place', dest='sample_place', action='store_true', default=False)
    parser.add_argument('--log_suffix', action='store', type=str, default=None)

    parser.add_argument('--feat_backbone', action='store', type=str, default='clip')
    parser.add_argument('--sample_num', action='store', type=int, default=500)
    parser.add_argument('--num_obj', action='store', type=int, default=8)
    parser.add_argument('--num_episode', action='store', type=int, default=5000)

    # Model loading options (for future use, place currently uses random selection)
    parser.add_argument('--load_model', action='store_true', default=False,
                        help='Load trained model for action selection')
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to trained model checkpoint')

    # Transformer model parameters (required when loading model)
    parser.add_argument('--ratio', type=float, default=0.2)
    parser.add_argument('--fusion_sa', action='store_true', default=False)
    parser.add_argument('--layer_norm', action='store_true', default=False)
    parser.add_argument('--lang_emb', action='store_true', default=False)
    parser.add_argument('--lang_enc', type=str, default='longclip')
    parser.add_argument('--use_rope', action='store_true', default=False)
    parser.add_argument('--no_feat_rope', action='store_true', default=False)
    parser.add_argument('--no_rgb_feat', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--adaptive', action='store_true', default=False)
    parser.add_argument('--adaptive_type', type=str, default='policy')
    parser.add_argument('--task_emb', action='store_true', default=False)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=384)

    # VLA recording options
    parser.add_argument('--record_vla', action='store_true', default=False,
                        help='Enable VLA recording to LeRobot dataset')
    parser.add_argument('--vla_output', type=str, default='data/vla_place',
                        help='Output directory for VLA dataset')
    parser.add_argument('--vla_repo_id', type=str, default='local/place_vla',
                        help='Repository ID for LeRobot dataset')
    parser.add_argument('--vla_fps', type=int, default=30,
                        help='Recording FPS for VLA data')
    parser.add_argument('--vla_image_size', type=int, nargs=2, default=[480, 640],
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Image size for VLA recording (default: 480 640)')
    parser.add_argument('--vla_cameras', type=str, default='front,overview',
                        help='Comma-separated cameras: front,left,right,top,side_left,side_right,overview (default: front,overview)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    # set device and seed
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Set task_num for model compatibility
    args.task_num = 2 if args.task_emb else None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # parameters
    num_obj = args.num_obj
    num_episode = args.num_episode

    # load environment
    env = Environment(gui=False)
    env.seed(args.seed)
    # env_sim = Environment(gui=False)
    # load logger
    logger = Logger(suffix=args.log_suffix)
    # load placenet
    placenet = Placenet()
    # load feature field builer
    field_builer = FeatureField(num_cam=3, query_threshold=[0.4], grid_size=0.004, boundaries=WORKSPACE_LIMITS, feat_backbone=args.feat_backbone, device=args.device)

    extrinsics, intrinsics = utils.get_camera_configs(env)
    logger.save_camera_configs(extrinsics, intrinsics)

    data = CLIPActionDataset()

    # VLA recorder initialization
    vla_recorder = None
    if args.record_vla:
        from helpers.vla_recorder import VLARecorder
        cameras = [c.strip() for c in args.vla_cameras.split(',')]
        vla_recorder = VLARecorder(
            output_dir=args.vla_output,
            repo_id=args.vla_repo_id,
            fps=args.vla_fps,
            image_size=tuple(args.vla_image_size),
            cameras=cameras,
        )
        env.set_frame_recorder(vla_recorder.record_frame, fps=args.vla_fps, cameras=cameras)

    iteration = 0
    updates = 0

    # collect data with clip agent
    for episode in range(num_episode):
        episode_reward = 0
        episode_steps = 0
        done = False
        reset = False

        while not reset:
            env.reset()
            # env_sim.reset()
            if episode < 400:
                warmup_num_obj = 5
                _, reset = env.add_objects_for_place(warmup_num_obj, WORKSPACE_LIMITS)
            else:
                _, reset = env.add_objects_for_place(num_obj, WORKSPACE_LIMITS)
            # reset &= env_sim.add_objects(num_obj, WORKSPACE_LIMITS)

        while not done:
            # remove labels and ids of objects out of workspace
            out_of_workspace = []
            for obj_id in env.obj_ids["rigid"]:
                pos, _, _ = env.obj_info(obj_id)
                if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
                    or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
                    print("\033[031m Delete objects out of workspace!\033[0m")
                    env.remove_object_id(obj_id)

            color_images, depth_images, pcd = utils.get_multi_view_images_w_pointcloud(env, visualize=args.visualize)

            color_heightmap, depth_heightmap, mask_heightmap = utils.get_true_heightmap(env)
            bbox_ids, bbox_images, bbox_sizes, bbox_centers, bbox_positions = utils.get_true_bboxes(env, color_heightmap, depth_heightmap, mask_heightmap)
            bbox_ids, remain_bbox_images, bbox_sizes, bbox_centers, bbox_positions = utils.preprocess_bboxes(bbox_ids, bbox_images, bbox_sizes, bbox_centers, bbox_positions)

            # guarentee all objects are detected when collecting data
            if len(remain_bbox_images) != len(bbox_images): # or len(remain_bbox_images) < len(env.obj_ids["rigid"]):
                print("\033[031m Bad detection of the scene!\033[0m")
                break
            # guarentee there are objects with labels
            if len(env.obj_labels) == 0:
                print("\033[031m No labeled objects in the scene!\033[0m")
                break

            lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, valid_mask = env.generate_place_lang_goal(bbox_ids, bbox_centers, bbox_sizes)
            if lang_goal is None:
                print("\033[031m Nonvalid scene!\033[0m")
                break
            else:
                print(f"\033[032m Reset environment of episode {episode}, language goal {lang_goal}\033[0m")

            # Start VLA recording for this episode (after lang_goal is known)
            if vla_recorder is not None:
                vla_recorder.start_episode(task=lang_goal)

            # check if one of the target objects is in the workspace:
            out_of_workspace = []
            for obj_id in env.reference_obj_ids:
                pos, _, _ = env.obj_info(obj_id)
                if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
                    or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
                    out_of_workspace.append(obj_id)

            if len(env.reference_obj_ids) > 0 and len(out_of_workspace) == len(env.reference_obj_ids):
                print("\033[031m Target objects are not in the scene!\033[0m")
                break

            # graspnet
            # grasp pose set is all zero when placing
            dump_action_num = 10
            grasp_pose_set = [np.zeros(7) for _ in range(dump_action_num)]

            # placenet, generate all feasible places
            all_place_valid_mask = utils.generate_all_place_dist(ref_obj_ids[0], env.obj_labels[ref_obj_ids[0]][0], ref_regions[0], valid_mask, env.obj_labels, bbox_ids, bbox_centers, bbox_sizes)
            # place_pixels, place_pose_set, _, valid_places_list = placenet.place_generation_return_gt(depth_heightmap, bbox_centers, bbox_sizes, ref_obj_centers, ref_regions, all_place_valid_mask, grasped_obj_size=None)
            place_pixels, place_pose_set, _, valid_places_list = placenet.place_generation_return_gt(depth_heightmap, bbox_centers, bbox_sizes, ref_obj_centers, ref_regions, all_place_valid_mask, grasped_obj_size=None, sample_num_each_object=3)

            if len(valid_places_list) == 0:
                print("\033[031m Nonvalid place in the scene!\033[0m")
                break

            pts = utils.generate_points_for_feature_extraction(pcd, cut_table=False, visualize=args.visualize)

            # generated feature
            feature_list = ['clip_feats', 'clip_sims']
            # pts, feat_dict = field_builer.generate_feature_field(color_images, depth_images, extrinsics, intrinsics, lang_goal, feature_list, args.visualize)  # 7.33s
            pts, feat_dict = field_builer.generate_feature_field(color_images, depth_images, extrinsics, intrinsics, lang_goal, feature_list, pts, last_text_feature=False, visualize=args.visualize)  # 2s

            # preprocess
            sampled_pts, sampled_clip_feats, sampled_clip_sims, grasps, grasp_pose_set, places, place_pose_set = utils.preprocess_pp(pts, feat_dict, grasp_pose_set, place_pose_set, sample_grasp=args.sample_grasp, \
                                                                                                            sample_place=args.sample_place, sample_num=args.sample_num, visualize=args.visualize)


            action_idx = np.random.choice(valid_places_list, size=1)[0]

            if action_idx >= 0:
                action = place_pose_set[action_idx]
            else:
                print("\033[031m Nonvalid scene!\033[0m")
                break

            # Set action for VLA recording before executing
            if vla_recorder is not None:
                vla_recorder.set_action(action, attempt_idx=episode_steps)

            # reward, done = env.step(action)
            reward = 2
            done = True

            # Update step result for VLA recording
            if vla_recorder is not None:
                vla_recorder.update_step_result(reward=reward, success=done)

            episode_steps += 1
            iteration += 1
            # episode_reward += reward
            print("\033[034m Episode: {}, total numsteps: {}\033[0m".format(episode, iteration, done))

            # data collection
            # for unified
            data.add(episode, episode_steps, lang_goal, sampled_pts.detach().cpu().numpy()[0], sampled_clip_feats.detach().cpu().numpy()[0], sampled_clip_sims.detach().cpu().numpy()[0], places.detach().cpu().numpy()[0], action_idx, reward, done)

            if done: break

        # End VLA recording for this episode
        if vla_recorder is not None:
            vla_recorder.end_episode(success=done, total_reward=episode_reward, num_attempts=episode_steps)

        if (episode + 1) % 5000 == 0:
            timestamp = time.time()
            timestamp_value = datetime.datetime.fromtimestamp(timestamp)
            name = 'train_' + timestamp_value.strftime('%Y_%m_%d_%H_%M_%S_') + str(iteration) + '.npy'
            data.save(name)

        logger.episode_success_logs.append(done)
        logger.write_to_log('episode_success', logger.episode_success_logs)
        print("\033[034m Episode: {}, total numsteps: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(episode, iteration, episode_steps, round(episode_reward, 2), done))

    # Finalize VLA recording
    if vla_recorder is not None:
        vla_recorder.finalize()
        print(f"VLA dataset saved to {args.vla_output}")
