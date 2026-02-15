#!/usr/bin/env python3
"""Render and validate textures for all object assets.

Default behavior matches benchmark-style loading (no forced MTL color override).

Covers:
- URDF-backed objects in assets/simplified_objects/*.urdf and assets/unseen_objects/*.urdf
- Folder-backed simplified objects without a top-level URDF (e.g. 000, 001, ...)
"""

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import pybullet as p


@dataclass
class RenderTarget:
    key: str
    kind: str  # 'urdf' or 'mesh'
    urdf_rel: Optional[str] = None
    visual_mesh_abs: Optional[str] = None
    collision_mesh_abs: Optional[str] = None


def find_mtl_for_obj(obj_path: str) -> Optional[str]:
    if not os.path.exists(obj_path):
        return None
    try:
        with open(obj_path, "r", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.lower().startswith("mtllib "):
                    mtl_name = s.split(None, 1)[1].strip()
                    mtl_path = os.path.normpath(os.path.join(os.path.dirname(obj_path), mtl_name))
                    return mtl_path if os.path.exists(mtl_path) else None
    except OSError:
        return None
    return None


def parse_map_kd(mtl_path: Optional[str]) -> Tuple[str, str]:
    """Return (raw_map_kd_value, existing_abs_texture_path_or_empty)."""
    if not mtl_path or not os.path.exists(mtl_path):
        return "", ""
    try:
        with open(mtl_path, "r", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.lower().startswith("map_kd "):
                    raw_name = s.split(None, 1)[1].strip()
                    tex_abs = os.path.normpath(os.path.join(os.path.dirname(mtl_path), raw_name))
                    return raw_name, tex_abs if os.path.exists(tex_abs) else ""
    except OSError:
        return "", ""
    return "", ""


def collect_targets(root: str) -> List[RenderTarget]:
    targets: List[RenderTarget] = []

    for urdf in sorted(glob.glob(os.path.join(root, "assets/simplified_objects/*.urdf"))):
        targets.append(RenderTarget(key=os.path.relpath(urdf, root), kind="urdf", urdf_rel=os.path.relpath(urdf, root)))

    for urdf in sorted(glob.glob(os.path.join(root, "assets/unseen_objects/*.urdf"))):
        base = os.path.basename(urdf)
        if base.startswith("."):
            continue
        targets.append(RenderTarget(key=os.path.relpath(urdf, root), kind="urdf", urdf_rel=os.path.relpath(urdf, root)))

    simp_dir = os.path.join(root, "assets/simplified_objects")
    urdf_ids = {
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(os.path.join(simp_dir, "*.urdf"))
    }
    for entry in sorted(os.listdir(simp_dir)):
        if not re.fullmatch(r"\d{3}", entry):
            continue
        if entry in urdf_ids:
            continue
        obj_dir = os.path.join(simp_dir, entry)
        visual_candidates = [
            os.path.join(obj_dir, "textured_simplified.obj"),
            os.path.join(obj_dir, "textured.obj"),
            os.path.join(obj_dir, "collision.obj"),
        ]
        collision_candidates = [
            os.path.join(obj_dir, "textured_simplified_vhacd.obj"),
            os.path.join(obj_dir, "textured_vhacd.obj"),
            os.path.join(obj_dir, "collision_vhacd.obj"),
        ]
        visual = next((path for path in visual_candidates if os.path.exists(path)), None)
        if visual is None:
            continue
        collision = next((path for path in collision_candidates if os.path.exists(path)), visual)
        targets.append(
            RenderTarget(
                key=f"assets/simplified_objects/{entry} (mesh)",
                kind="mesh",
                visual_mesh_abs=visual,
                collision_mesh_abs=collision,
            )
        )

    return targets


def _resolve_visual_obj(root: str, target: RenderTarget) -> str:
    if target.kind == "mesh":
        return target.visual_mesh_abs or ""

    urdf_abs = os.path.join(root, target.urdf_rel)
    text = open(urdf_abs, "r", errors="ignore").read()
    match = re.search(r'<mesh filename="([^"]+)"', text)
    if not match:
        return ""
    return os.path.normpath(os.path.join(os.path.dirname(urdf_abs), match.group(1)))


def render_target(
    root: str,
    target: RenderTarget,
    out_dir: str,
    image_size: int,
    renderer: int,
    urdf_flags: int,
) -> Tuple[float, float, float, float, str]:
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)

    if target.kind == "urdf":
        body_id = p.loadURDF(
            target.urdf_rel,
            [0, 0, 0],
            useFixedBase=True,
            flags=urdf_flags,
        )
    else:
        visual = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=target.visual_mesh_abs,
            meshScale=[1, 1, 1],
        )
        collision = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=target.collision_mesh_abs,
            meshScale=[1, 1, 1],
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0, 0, 0],
        )

    aabb_min, aabb_max = p.getAABB(body_id)
    mn = np.array(aabb_min)
    mx = np.array(aabb_max)
    center = (mn + mx) / 2
    ext = np.maximum(mx - mn, 1e-3)
    radius = float(np.linalg.norm(ext))
    dist = max(0.25, radius * 1.8)

    eye = (center + np.array([dist, dist, dist * 0.8])).tolist()
    view = p.computeViewMatrix(eye, center.tolist(), [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(60, 1.0, 0.01, 5.0)
    _, _, rgba, _, seg = p.getCameraImage(
        image_size,
        image_size,
        view,
        proj,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        renderer=renderer,
    )

    arr = np.array(rgba, dtype=np.uint8).reshape(image_size, image_size, 4)[:, :, :3]
    seg = np.array(seg, dtype=np.int32).reshape(image_size, image_size)

    luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    full_mean_luma = float(luma.mean())
    full_black_frac = float((luma < 15).mean())

    # Segment object pixels robustly (base-link encoding includes object id in lower 24 bits).
    obj_mask = (seg == body_id) | ((seg & ((1 << 24) - 1)) == body_id)
    if obj_mask.any():
        obj_luma = luma[obj_mask]
        obj_mean_luma = float(obj_luma.mean())
        obj_black_frac = float((obj_luma < 15).mean())
    else:
        obj_mean_luma = full_mean_luma
        obj_black_frac = full_black_frac

    png_name = target.key.replace("/", "__").replace(" ", "_") + ".png"
    Image.fromarray(arr).save(os.path.join(out_dir, png_name))
    return full_mean_luma, full_black_frac, obj_mean_luma, obj_black_frac, png_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        default="debug_runs/texture_audit_all",
        help="Output root directory (relative to repo root).",
    )
    parser.add_argument(
        "--renderer",
        choices=["opengl", "tiny"],
        default="opengl",
        help="Renderer to use for audit images.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square render size in pixels.",
    )
    parser.add_argument(
        "--force-mtl-colors",
        action="store_true",
        help="Force URDF_USE_MATERIAL_COLORS_FROM_MTL (debug mode; not benchmark behavior).",
    )
    parser.add_argument(
        "--no-sleep-flag",
        action="store_true",
        help="Do not set URDF_ENABLE_SLEEPING for URDF loads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_root = os.path.join(root, args.out_root)
    render_dir = os.path.join(out_root, "renders")
    os.makedirs(render_dir, exist_ok=True)

    renderer = p.ER_BULLET_HARDWARE_OPENGL if args.renderer == "opengl" else p.ER_TINY_RENDERER

    urdf_flags = 0
    if not args.no_sleep_flag:
        urdf_flags |= p.URDF_ENABLE_SLEEPING
    if args.force_mtl_colors:
        urdf_flags |= p.URDF_USE_MATERIAL_COLORS_FROM_MTL

    targets = collect_targets(root)

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(root)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    rows = []
    for target in targets:
        try:
            visual_obj = _resolve_visual_obj(root, target)
            visual_rel = os.path.relpath(visual_obj, root) if visual_obj else ""

            mtl_rel = ""
            map_kd = ""
            tex_rel = ""
            tex_exists = ""
            tex_mean_luma = ""

            if visual_obj.endswith(".obj") and os.path.exists(visual_obj):
                mtl_abs = find_mtl_for_obj(visual_obj)
                if mtl_abs:
                    mtl_rel = os.path.relpath(mtl_abs, root)
                    map_kd, tex_abs = parse_map_kd(mtl_abs)
                    if map_kd:
                        if tex_abs:
                            tex_rel = os.path.relpath(tex_abs, root)
                            tex_exists = "1"
                            try:
                                tex_img = np.array(Image.open(tex_abs).convert("RGB"), dtype=np.uint8)
                                tex_luma = 0.2126 * tex_img[:, :, 0] + 0.7152 * tex_img[:, :, 1] + 0.0722 * tex_img[:, :, 2]
                                tex_mean_luma = f"{float(tex_luma.mean()):.2f}"
                            except Exception:
                                tex_mean_luma = ""
                        else:
                            tex_exists = "0"

            full_mean, full_black, obj_mean, obj_black, png_name = render_target(
                root=root,
                target=target,
                out_dir=render_dir,
                image_size=args.image_size,
                renderer=renderer,
                urdf_flags=urdf_flags,
            )

            # Flag as suspicious when texture is expected but object renders nearly black.
            suspicious_black = "0"
            if tex_exists == "1" and obj_mean < 12.0:
                suspicious_black = "1"
            if tex_exists == "1" and tex_mean_luma:
                if float(tex_mean_luma) > 25.0 and obj_mean < 15.0:
                    suspicious_black = "1"

            rows.append([
                target.key,
                target.kind,
                visual_rel,
                mtl_rel,
                map_kd,
                tex_rel,
                tex_exists,
                tex_mean_luma,
                f"{full_mean:.2f}",
                f"{full_black:.4f}",
                f"{obj_mean:.2f}",
                f"{obj_black:.4f}",
                suspicious_black,
                png_name,
                "",
            ])
        except Exception as exc:  # pylint: disable=broad-except
            rows.append([target.key, target.kind, "", "", "", "", "", "", "", "", "", "", "", "", str(exc)])

    p.disconnect()

    csv_path = os.path.join(out_root, "audit_all.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "target",
            "kind",
            "visual_obj",
            "mtl",
            "map_kd",
            "texture",
            "texture_exists",
            "texture_mean_luma",
            "render_mean_luma",
            "render_black_frac_lt15",
            "obj_mean_luma",
            "obj_black_frac_lt15",
            "suspicious_black",
            "render_png",
            "error",
        ])
        writer.writerows(rows)

    total = len(rows)
    errors = sum(1 for row in rows if row[-1])
    missing_map_kd = sum(1 for row in rows if row[2] and row[3] and not row[4])
    missing_tex_refs = sum(1 for row in rows if row[2] and row[4] and row[6] == "0")
    suspicious = sum(1 for row in rows if row[12] == "1")

    print(
        f"targets={total} errors={errors} missing_map_kd={missing_map_kd} missing_tex_refs={missing_tex_refs} "
        f"suspicious_black={suspicious} flags={urdf_flags} renderer={args.renderer}"
    )
    print(f"csv={csv_path}")
    print(f"renders={render_dir}")


if __name__ == "__main__":
    main()
