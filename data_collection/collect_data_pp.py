import argparse
import json
import os
import random
import shutil
import datetime
import warnings

# Suppress pydantic warnings from lerobot
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import numpy as np
import torch
import pybullet as pb

import utils.utils as utils
from env.constants import WORKSPACE_LIMITS
from env.environment_sim import Environment
from helpers.logger import Logger
from helpers.datasets import CLIPActionDataset
from action_generator.grasp_detetor import Graspnet
from action_generator.place_generator import Placenet
from feature_extractor.feature_field_builder import FeatureField
from models.clip_agent import CLIPGrasp
from models.mb_agent import MBGrasp


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=234, help="random seed")
    parser.add_argument("--num_episode_grasp", type=int, default=5000)
    parser.add_argument("--num_episode_place", type=int, default=5000)
    parser.add_argument("--num_obj_grasp", type=int, default=15)
    parser.add_argument("--num_obj_place", type=int, default=8)
    parser.add_argument("--sample_num", type=int, default=500)
    parser.add_argument("--max_episode_step", type=int, default=8)
    parser.add_argument("--feat_backbone", type=str, default="clip")
    parser.add_argument("--agent", type=str, default="mb", choices=["mb", "clip"])
    parser.add_argument("--sample_grasp", action="store_true", default=False)
    parser.add_argument("--sample_place", action="store_true", default=False)
    parser.add_argument("--output_path", type=str, default="data/a2_pp_data.npy")
    parser.add_argument("--record_dir", type=str, default="data/a2_pp_frames")
    parser.add_argument("--no_record", action="store_true", default=False)
    parser.add_argument("--log_suffix", type=str, default="recreate-pp")
    parser.add_argument("--visualize", action="store_true", default=False)

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
    parser.add_argument('--vla_output', type=str, default='data/vla_pp',
                        help='Output directory for VLA dataset')
    parser.add_argument('--vla_repo_id', type=str, default='local/pp_vla',
                        help='Repository ID for LeRobot dataset')
    parser.add_argument('--vla_fps', type=int, default=30,
                        help='Recording FPS for VLA data')
    parser.add_argument('--vla_image_size', type=int, nargs=2, default=[480, 640],
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Image size for VLA recording (default: 480 640)')
    parser.add_argument('--vla_cameras', type=str, default='front,overview',
                        help='Comma-separated cameras: front,left,right,top,side_left,side_right,overview (default: front,overview)')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_robot_state(env):
    ee_pos, ee_quat = env.get_link_pose(env.ur5e, env.ur5e_ee_id)
    joint_positions = [
        pb.getJointState(env.ur5e, j, physicsClientId=env._client_id)[0]
        for j in env.ur5e_joints
    ]
    gripper_angle = pb.getJointState(
        env.ee, env.gripper_main_joint, physicsClientId=env._client_id
    )[0]
    return {
        "ee_pos": np.asarray(ee_pos, dtype=np.float32),
        "ee_quat": np.asarray(ee_quat, dtype=np.float32),
        "joint_positions": np.asarray(joint_positions, dtype=np.float32),
        "gripper_angle": float(gripper_angle),
    }


class EpisodeRecorder:
    def __init__(self, root_dir, task, episode, lang_goal):
        self.task = task
        self.episode = int(episode)
        self.lang_goal = lang_goal
        self.step = 0
        self.root_dir = root_dir
        self.tmp_dir = os.path.join(root_dir, "_tmp", task, f"episode_{episode:05d}")
        self.final_dir = os.path.join(root_dir, task, f"episode_{episode:05d}")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def save_step(self, color_images, depth_images, robot_state, action, action_idx):
        step_path = os.path.join(self.tmp_dir, f"step_{self.step:04d}.npz")
        np.savez_compressed(
            step_path,
            color=color_images,
            depth=depth_images,
            ee_pos=robot_state["ee_pos"],
            ee_quat=robot_state["ee_quat"],
            joints=robot_state["joint_positions"],
            gripper_angle=np.asarray(robot_state["gripper_angle"], dtype=np.float32),
            action=np.asarray(action, dtype=np.float32),
            action_idx=np.asarray(action_idx, dtype=np.int32),
        )
        self.step += 1

    def commit(self):
        os.makedirs(os.path.dirname(self.final_dir), exist_ok=True)
        if os.path.exists(self.final_dir):
            shutil.rmtree(self.final_dir)
        os.replace(self.tmp_dir, self.final_dir)
        meta = {
            "task": self.task,
            "episode": self.episode,
            "lang_goal": self.lang_goal,
            "num_steps": self.step,
        }
        with open(os.path.join(self.final_dir, "episode.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def discard(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)


def save_camera_configs(record_dir, extrinsics, intrinsics):
    if record_dir is None:
        return
    os.makedirs(record_dir, exist_ok=True)
    np.save(os.path.join(record_dir, "camera_extrinsics.npy"), extrinsics)
    np.save(os.path.join(record_dir, "camera_intrinsics.npy"), intrinsics)


def collect_grasp(args, dataset, record_dir, vla_recorder=None, trained_agent=None):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.device = device

    env = Environment(gui=False)
    env.seed(args.seed)
    logger = Logger(suffix=f"{args.log_suffix}-grasp")
    graspnet = Graspnet()
    field_builder = FeatureField(
        num_cam=3,
        query_threshold=[0.4],
        grid_size=0.004,
        boundaries=WORKSPACE_LIMITS,
        feat_backbone=args.feat_backbone,
        device=device,
    )

    # Use trained agent if provided, otherwise use mb/clip agent
    if trained_agent is not None:
        agent = trained_agent
        use_trained_model = True
    elif args.agent == "clip":
        agent = CLIPGrasp(args)
        use_trained_model = False
    else:
        agent = MBGrasp(args)
        use_trained_model = False

    # Set up VLA recording if provided
    if vla_recorder is not None:
        cameras = [c.strip() for c in args.vla_cameras.split(',')]
        env.set_frame_recorder(vla_recorder.record_frame, fps=args.vla_fps, cameras=cameras)

    extrinsics, intrinsics = utils.get_camera_configs(env)
    logger.save_camera_configs(extrinsics, intrinsics)
    save_camera_configs(record_dir, extrinsics, intrinsics)

    iteration = 0

    for episode in range(args.num_episode_grasp):
        episode_reward = 0
        episode_steps = 0
        done = False
        reset = False

        while not reset:
            env.reset()
            lang_goal = env.generate_lang_goal()
            if episode < 400:
                warmup_num_obj = 8
                reset = env.add_objects(warmup_num_obj, WORKSPACE_LIMITS)
            else:
                reset = env.add_objects(args.num_obj_grasp, WORKSPACE_LIMITS)
            print(
                f"\033[032m Reset environment of episode {episode}, language goal {lang_goal}\033[0m"
            )

        recorder = None
        if record_dir is not None:
            recorder = EpisodeRecorder(record_dir, "grasp", episode, lang_goal)

        # Start VLA recording for this episode
        if vla_recorder is not None:
            vla_recorder.start_episode(task=lang_goal)

        while not done:
            out_of_workspace = []
            for obj_id in env.target_obj_ids:
                pos, _, _ = env.obj_info(obj_id)
                if (
                    pos[0] < WORKSPACE_LIMITS[0][0]
                    or pos[0] > WORKSPACE_LIMITS[0][1]
                    or pos[1] < WORKSPACE_LIMITS[1][0]
                    or pos[1] > WORKSPACE_LIMITS[1][1]
                ):
                    out_of_workspace.append(obj_id)

            if len(out_of_workspace) == len(env.target_obj_ids):
                print("\033[031m Target objects are not in the scene!\033[0m")
                break

            color_images, depth_images, pcd = utils.get_multi_view_images_w_pointcloud(
                env, visualize=args.visualize
            )

            with torch.no_grad():
                grasp_pose_set, _, _ = graspnet.grasp_detection(
                    pcd, env.get_true_object_poses(), visualize=args.visualize
                )
            print("Number of grasping poses", len(grasp_pose_set))
            if len(grasp_pose_set) == 0:
                break

            pts = utils.generate_points_for_feature_extraction(pcd, visualize=args.visualize)
            feature_list = ["clip_feats", "clip_sims"]
            pts, feat_dict = field_builder.generate_feature_field(
                color_images,
                depth_images,
                extrinsics,
                intrinsics,
                lang_goal,
                feature_list,
                pts,
                last_text_feature=False,
                visualize=args.visualize,
            )

            sampled_pts, sampled_clip_feats, sampled_clip_sims, grasps, grasp_pose_set = (
                utils.preprocess_pp_unified(
                    pts,
                    feat_dict,
                    grasp_pose_set,
                    sample_action=args.sample_grasp,
                    sample_num=args.sample_num,
                    visualize=args.visualize,
                )
            )

            if use_trained_model:
                # Use trained model for action selection
                with torch.no_grad():
                    logits, action_idx = agent.select_action(
                        sampled_pts, sampled_clip_feats, sampled_clip_sims, grasps
                    )
            elif args.agent == "clip":
                if len(grasp_pose_set) == 1:
                    action_idx = 0
                else:
                    with torch.no_grad():
                        action_idx = agent.select_action_greedy(
                            sampled_pts, sampled_clip_sims, grasps
                        )
            else:
                action_idx = agent.select_action_greedy(
                    grasp_pose_set, env.get_target_object_poses()
                )

            action = grasp_pose_set[action_idx]

            if recorder is not None:
                robot_state = get_robot_state(env)
                recorder.save_step(
                    color_images, depth_images, robot_state, action, action_idx
                )

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
            print(
                "\033[034m Episode: {}, total numsteps: {}, reward: {}\033[0m".format(
                    episode, iteration, round(reward, 2), done
                )
            )

            if done:
                dataset.add(
                    episode,
                    episode_steps,
                    lang_goal,
                    sampled_pts.detach().cpu().numpy()[0],
                    sampled_clip_feats.detach().cpu().numpy()[0],
                    sampled_clip_sims.detach().cpu().numpy()[0],
                    grasps.detach().cpu().numpy()[0],
                    action_idx,
                    reward,
                    done,
                )

            logger.reward_logs.append(reward)
            logger.executed_action_logs.append(action)
            logger.write_to_log("reward", logger.reward_logs)
            logger.write_to_log("executed_action", logger.executed_action_logs)

            if done or episode_steps == args.max_episode_step:
                break

        # End VLA recording for this episode
        if vla_recorder is not None:
            vla_recorder.end_episode(success=done, total_reward=episode_reward, num_attempts=episode_steps)

        if recorder is not None:
            if done:
                recorder.commit()
            else:
                recorder.discard()

        logger.episode_reward_logs.append(episode_reward)
        logger.episode_step_logs.append(episode_steps)
        logger.episode_success_logs.append(done)
        logger.write_to_log("episode_reward", logger.episode_reward_logs)
        logger.write_to_log("episode_step", logger.episode_step_logs)
        logger.write_to_log("episode_success", logger.episode_success_logs)
        print(
            "\033[034m Episode: {}, total numsteps: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(
                episode, iteration, episode_steps, round(episode_reward, 2), done
            )
        )


def collect_place(args, dataset, record_dir, vla_recorder=None, trained_agent=None):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Environment(gui=False)
    env.seed(args.seed)
    logger = Logger(suffix=f"{args.log_suffix}-place")
    placenet = Placenet()
    field_builder = FeatureField(
        num_cam=3,
        query_threshold=[0.4],
        grid_size=0.004,
        boundaries=WORKSPACE_LIMITS,
        feat_backbone=args.feat_backbone,
        device=device,
    )

    # Note: trained_agent currently not used for place (uses random selection)
    # but kept for future expansion
    use_trained_model = trained_agent is not None

    # Set up VLA recording if provided
    if vla_recorder is not None:
        cameras = [c.strip() for c in args.vla_cameras.split(',')]
        env.set_frame_recorder(vla_recorder.record_frame, fps=args.vla_fps, cameras=cameras)

    extrinsics, intrinsics = utils.get_camera_configs(env)
    logger.save_camera_configs(extrinsics, intrinsics)
    save_camera_configs(record_dir, extrinsics, intrinsics)

    iteration = 0

    for episode in range(args.num_episode_place):
        episode_reward = 0
        episode_steps = 0
        done = False
        reset = False

        while not reset:
            env.reset()
            if episode < 400:
                warmup_num_obj = 5
                _, reset = env.add_objects_for_place(warmup_num_obj, WORKSPACE_LIMITS)
            else:
                _, reset = env.add_objects_for_place(args.num_obj_place, WORKSPACE_LIMITS)

        recorder = None

        while not done:
            out_of_workspace = []
            for obj_id in env.obj_ids["rigid"]:
                pos, _, _ = env.obj_info(obj_id)
                if (
                    pos[0] < WORKSPACE_LIMITS[0][0]
                    or pos[0] > WORKSPACE_LIMITS[0][1]
                    or pos[1] < WORKSPACE_LIMITS[1][0]
                    or pos[1] > WORKSPACE_LIMITS[1][1]
                ):
                    print("\033[031m Delete objects out of workspace!\033[0m")
                    env.remove_object_id(obj_id)

            color_images, depth_images, pcd = utils.get_multi_view_images_w_pointcloud(
                env, visualize=args.visualize
            )

            color_heightmap, depth_heightmap, mask_heightmap = utils.get_true_heightmap(env)
            bbox_ids, bbox_images, bbox_sizes, bbox_centers, bbox_positions = utils.get_true_bboxes(
                env, color_heightmap, depth_heightmap, mask_heightmap
            )
            bbox_ids, remain_bbox_images, bbox_sizes, bbox_centers, bbox_positions = (
                utils.preprocess_bboxes(
                    bbox_ids, bbox_images, bbox_sizes, bbox_centers, bbox_positions
                )
            )

            if len(remain_bbox_images) != len(bbox_images):
                print("\033[031m Bad detection of the scene!\033[0m")
                break
            if len(env.obj_labels) == 0:
                print("\033[031m No labeled objects in the scene!\033[0m")
                break

            lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, valid_mask = (
                env.generate_place_lang_goal(bbox_ids, bbox_centers, bbox_sizes)
            )
            if lang_goal is None:
                print("\033[031m Nonvalid scene!\033[0m")
                break
            print(
                f"\033[032m Reset environment of episode {episode}, language goal {lang_goal}\033[0m"
            )

            if recorder is None and record_dir is not None:
                recorder = EpisodeRecorder(record_dir, "place", episode, lang_goal)

            # Start VLA recording for this episode (after lang_goal is known)
            if vla_recorder is not None and not vla_recorder.is_recording:
                vla_recorder.start_episode(task=lang_goal)

            out_of_workspace = []
            for obj_id in env.reference_obj_ids:
                pos, _, _ = env.obj_info(obj_id)
                if (
                    pos[0] < WORKSPACE_LIMITS[0][0]
                    or pos[0] > WORKSPACE_LIMITS[0][1]
                    or pos[1] < WORKSPACE_LIMITS[1][0]
                    or pos[1] > WORKSPACE_LIMITS[1][1]
                ):
                    out_of_workspace.append(obj_id)

            if len(env.reference_obj_ids) > 0 and len(out_of_workspace) == len(
                env.reference_obj_ids
            ):
                print("\033[031m Target objects are not in the scene!\033[0m")
                break

            dump_action_num = 10
            grasp_pose_set = [np.zeros(7) for _ in range(dump_action_num)]

            all_place_valid_mask = utils.generate_all_place_dist(
                ref_obj_ids[0],
                env.obj_labels[ref_obj_ids[0]][0],
                ref_regions[0],
                valid_mask,
                env.obj_labels,
                bbox_ids,
                bbox_centers,
                bbox_sizes,
            )
            place_pixels, place_pose_set, _, valid_places_list = (
                placenet.place_generation_return_gt(
                    depth_heightmap,
                    bbox_centers,
                    bbox_sizes,
                    ref_obj_centers,
                    ref_regions,
                    all_place_valid_mask,
                    grasped_obj_size=None,
                    sample_num_each_object=3,
                )
            )

            if len(valid_places_list) == 0:
                print("\033[031m Nonvalid place in the scene!\033[0m")
                break

            pts = utils.generate_points_for_feature_extraction(
                pcd, cut_table=False, visualize=args.visualize
            )
            feature_list = ["clip_feats", "clip_sims"]
            pts, feat_dict = field_builder.generate_feature_field(
                color_images,
                depth_images,
                extrinsics,
                intrinsics,
                lang_goal,
                feature_list,
                pts,
                last_text_feature=False,
                visualize=args.visualize,
            )

            (
                sampled_pts,
                sampled_clip_feats,
                sampled_clip_sims,
                grasps,
                grasp_pose_set,
                places,
                place_pose_set,
            ) = utils.preprocess_pp(
                pts,
                feat_dict,
                grasp_pose_set,
                place_pose_set,
                sample_grasp=args.sample_grasp,
                sample_place=args.sample_place,
                sample_num=args.sample_num,
                visualize=args.visualize,
            )

            action_idx = np.random.choice(valid_places_list, size=1)[0]
            if action_idx < 0:
                print("\033[031m Nonvalid scene!\033[0m")
                break

            action = place_pose_set[action_idx]

            if recorder is not None:
                robot_state = get_robot_state(env)
                recorder.save_step(
                    color_images, depth_images, robot_state, action, action_idx
                )

            # Set action for VLA recording before executing
            if vla_recorder is not None:
                vla_recorder.set_action(action, attempt_idx=episode_steps)

            reward = 2
            done = True

            # Update step result for VLA recording
            if vla_recorder is not None:
                vla_recorder.update_step_result(reward=reward, success=done)

            episode_steps += 1
            iteration += 1
            episode_reward += reward
            print(
                "\033[034m Episode: {}, total numsteps: {}\033[0m".format(
                    episode, iteration
                )
            )

            dataset.add(
                episode,
                episode_steps,
                lang_goal,
                sampled_pts.detach().cpu().numpy()[0],
                sampled_clip_feats.detach().cpu().numpy()[0],
                sampled_clip_sims.detach().cpu().numpy()[0],
                places.detach().cpu().numpy()[0],
                action_idx,
                reward,
                done,
            )

            if done:
                break

        # End VLA recording for this episode
        if vla_recorder is not None:
            vla_recorder.end_episode(success=done, total_reward=episode_reward, num_attempts=episode_steps)

        if recorder is not None:
            if done:
                recorder.commit()
            else:
                recorder.discard()

        logger.episode_success_logs.append(done)
        logger.write_to_log("episode_success", logger.episode_success_logs)
        print(
            "\033[034m Episode: {}, total numsteps: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(
                episode, iteration, episode_steps, round(episode_reward, 2), done
            )
        )


def main():
    args = parse_args()
    record_dir = None if args.no_record else args.record_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    # Set task_num for model compatibility
    args.task_num = 2 if args.task_emb else None

    dataset = CLIPActionDataset()

    # Initialize VLA recorder if requested
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

    # Load trained model if requested
    trained_agent = None
    if args.load_model:
        from models.bc_agent import ViLGP3D
        from helpers.logger import Logger

        print(f"Loading trained model from {args.model_path}")
        trained_agent = ViLGP3D(action_dim=7, args=args)
        logger = Logger(suffix=args.log_suffix)
        logger.load_sl_checkpoint(trained_agent.vilg3d, args.model_path, evaluate=True)

    collect_grasp(args, dataset, record_dir, vla_recorder=vla_recorder, trained_agent=trained_agent)
    collect_place(args, dataset, record_dir, vla_recorder=vla_recorder, trained_agent=trained_agent)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(args.output_path, dataset.data)

    # Finalize VLA recording
    if vla_recorder is not None:
        vla_recorder.finalize()
        print(f"VLA dataset saved to {args.vla_output}")

    if record_dir is not None:
        meta = {
            "seed": args.seed,
            "num_episode_grasp": args.num_episode_grasp,
            "num_episode_place": args.num_episode_place,
            "num_obj_grasp": args.num_obj_grasp,
            "num_obj_place": args.num_obj_place,
            "sample_num": args.sample_num,
            "max_episode_step": args.max_episode_step,
            "output_path": args.output_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "total_samples": len(dataset.data["sequence"]),
        }
        with open(os.path.join(record_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
