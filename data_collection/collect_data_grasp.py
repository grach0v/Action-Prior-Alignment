import os
import time
import argparse
import warnings

# Suppress pydantic warnings from lerobot
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import numpy as np
import random
import datetime
from regex import F
import torch

import utils.utils as utils
from env.constants import WORKSPACE_LIMITS
from env.environment_sim import Environment
from helpers.logger import Logger
from action_generator.grasp_detetor import Graspnet
from feature_extractor.feature_field_builder import FeatureField
from models.clip_agent import CLIPGrasp
from models.mb_agent import MBGrasp
from helpers.dataset import CLIPActionDataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=234, metavar='N',
                    help='random seed (default: 1234)')

    parser.add_argument('--evaluate', dest='evaluate', action='store_true', default=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
    parser.add_argument('--sample_grasp', dest='sample_grasp', action='store_true', default=False)
    parser.add_argument('--log_suffix', action='store', type=str, default=None)

    parser.add_argument('--feat_backbone', action='store', type=str, default='clip')
    parser.add_argument('--agent', action='store', type=str, default='mb')
    parser.add_argument('--sample_num', action='store', type=int, default=500)
    parser.add_argument('--num_obj', action='store', type=int, default=15)
    parser.add_argument('--num_episode', action='store', type=int, default=5000)
    parser.add_argument('--max_episode_step', type=int, default=8)

    # Model loading options for faster data collection
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
    parser.add_argument('--vla_output', type=str, default='data/vla_grasp',
                        help='Output directory for VLA dataset')
    parser.add_argument('--vla_repo_id', type=str, default='local/grasp_vla',
                        help='Repository ID for LeRobot dataset')
    parser.add_argument('--vla_fps', type=int, default=30,
                        help='Recording FPS for VLA data (default: 30)')
    parser.add_argument('--vla_image_size', type=int, nargs=2, default=[480, 640],
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Image size stored in dataset (default: 480 640)')
    parser.add_argument('--vla_render_size', type=int, nargs=2, default=[480, 640],
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Render size (default: 480 640, needs GPU for 30 FPS)')
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
    # load graspnet
    graspnet = Graspnet()
    # load feature field builer
    field_builer = FeatureField(num_cam=3, query_threshold=[0.4], grid_size=0.004, boundaries=WORKSPACE_LIMITS, feat_backbone=args.feat_backbone, device=args.device)
    
    # Load trained model or use heuristic agent
    use_trained_model = False
    if args.load_model:
        from models.bc_agent import ViLGP3D
        print(f"Loading trained model from {args.model_path}")
        agent = ViLGP3D(action_dim=7, args=args)
        logger.load_sl_checkpoint(agent.vilg3d, args.model_path, evaluate=True)
        use_trained_model = True
    elif args.agent == "clip":
        # load clip agent
        agent = CLIPGrasp(args)
    elif args.agent == "mb":
        agent = MBGrasp(args)

    extrinsics, intrinsics = utils.get_camera_configs(env)
    logger.save_camera_configs(extrinsics, intrinsics)

    # data initialization
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
        render_size = tuple(args.vla_render_size) if args.vla_render_size else None
        env.set_frame_recorder(vla_recorder.record_frame, fps=args.vla_fps, cameras=cameras, render_size=render_size)

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
            lang_goal = env.generate_lang_goal()
            if episode < 400:
                warmup_num_obj = 8
                reset = env.add_objects(warmup_num_obj, WORKSPACE_LIMITS)
            else:
                reset = env.add_objects(num_obj, WORKSPACE_LIMITS)
            # reset &= env_sim.add_objects(num_obj, WORKSPACE_LIMITS)
            print(f"\033[032m Reset environment of episode {episode}, language goal {lang_goal}\033[0m")

        # Start VLA recording for this episode
        if vla_recorder is not None:
            vla_recorder.start_episode(task=lang_goal)

        while not done:
            # check if one of the target objects is in the workspace:
            out_of_workspace = []
            for obj_id in env.target_obj_ids:
                pos, _, _ = env.obj_info(obj_id)
                if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
                    or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
                    out_of_workspace.append(obj_id)

            if len(out_of_workspace) == len(env.target_obj_ids):
                print("\033[031m Target objects are not in the scene!\033[0m")
                break     
            
            color_images, depth_images, pcd = utils.get_multi_view_images_w_pointcloud(env, visualize=args.visualize)
            
            # graspnet
            # Note: simply replace object poses to object bounding boxes maybe ok
            with torch.no_grad():
                grasp_pose_set, grasp_pose_dict, _ = graspnet.grasp_detection(pcd, env.get_true_object_poses(), visualize=args.visualize)  # 1.19s
            print("Number of grasping poses", len(grasp_pose_set))
            if len(grasp_pose_set) == 0:
                break

            pts = utils.generate_points_for_feature_extraction(pcd, visualize=args.visualize)

            # generated feature
            # for vision-language tasks
            feature_list = ['clip_feats', 'clip_sims']
            
            pts, feat_dict = field_builer.generate_feature_field(color_images, depth_images, extrinsics, intrinsics, lang_goal, feature_list, pts, last_text_feature=False, visualize=args.visualize)  # 2s

            # preprocess
            sampled_pts, sampled_clip_feats, sampled_clip_sims, grasps, grasp_pose_set = utils.preprocess_pp_unified(pts, feat_dict, grasp_pose_set, sample_action=args.sample_grasp, sample_num=args.sample_num, visualize=args.visualize)

            if use_trained_model:
                # Use trained model for action selection
                with torch.no_grad():
                    logits, action_idx = agent.select_action(sampled_pts, sampled_clip_feats, sampled_clip_sims, grasps)
            elif args.agent == "clip":
                if len(grasp_pose_set) == 1:
                    action_idx = 0
                else:
                    with torch.no_grad():
                        action_idx = agent.select_action_greedy(sampled_pts, sampled_clip_sims, grasps)
            elif args.agent == "mb":
                action_idx = agent.select_action_greedy(grasp_pose_set, env.get_target_object_poses())

            action = grasp_pose_set[action_idx]

            # Set action for VLA recording before executing
            if vla_recorder is not None:
                vla_recorder.set_action(action, attempt_idx=episode_steps)

            reward, done = env.step(action)

            # Update step result for VLA recording
            if vla_recorder is not None:
                vla_recorder.update_step_result(reward=reward, success=done)

            episode_steps += 1
            iteration += 1
            episode_reward += reward
            print("\033[034m Episode: {}, total numsteps: {}, reward: {}\033[0m".format(episode, iteration, round(reward, 2), done))

            if done:
                # data collection
                data.add(episode, episode_steps, lang_goal, sampled_pts.detach().cpu().numpy()[0], sampled_clip_feats.detach().cpu().numpy()[0], sampled_clip_sims.detach().cpu().numpy()[0], grasps.detach().cpu().numpy()[0], action_idx, reward, done)

            # record
            logger.reward_logs.append(reward)
            logger.executed_action_logs.append(action)
            logger.write_to_log('reward', logger.reward_logs)
            logger.write_to_log('executed_action', logger.executed_action_logs)
            
            if done or episode_steps == args.max_episode_step:
                break

        # End VLA recording for this episode
        if vla_recorder is not None:
            vla_recorder.end_episode(success=done, total_reward=episode_reward, num_attempts=episode_steps)

        if (episode + 1) % 1000 == 0:
            timestamp = time.time()
            timestamp_value = datetime.datetime.fromtimestamp(timestamp)
            name = 'train_' + timestamp_value.strftime('%Y_%m_%d_%H_%M_%S_') + str(len(data.data['sequence'])) + '.npy'
            data.save(name)
            
        logger.episode_reward_logs.append(episode_reward)
        logger.episode_step_logs.append(episode_steps)
        logger.episode_success_logs.append(done)
        logger.write_to_log('episode_reward', logger.episode_reward_logs)
        logger.write_to_log('episode_step', logger.episode_step_logs)
        logger.write_to_log('episode_success', logger.episode_success_logs)
        print("\033[034m Episode: {}, total numsteps: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(episode, iteration, episode_steps, round(episode_reward, 2), done))

    # Finalize VLA recording
    if vla_recorder is not None:
        vla_recorder.finalize()
        print(f"VLA dataset saved to {args.vla_output}")