import datetime
import json
import os
import glob


def _truthy(value):
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_debug_level():
    raw = os.getenv("A2_DEBUG_LEVEL")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return 1 if _truthy(raw) else 0
    return 1 if (_truthy(os.getenv("A2_DEBUG_VERBOSE")) or _truthy(os.getenv("A2_DEBUG"))) else 0


def debug_enabled(level=1):
    return get_debug_level() >= level


def debug_log(component, message, payload=None, level=1):
    if not debug_enabled(level):
        return

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[A2DBG][{ts}][{component}] {message}"
    if payload is not None:
        try:
            line += " | " + json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            line += f" | {payload}"

    print(line, flush=True)

    log_file = os.getenv("A2_DEBUG_LOG_FILE")
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")


def texture_debug_enabled():
    return _truthy(os.getenv("A2_TEXTURE_DEBUG"))


def _decode_mesh_path(raw_path):
    if raw_path is None:
        return ""
    if isinstance(raw_path, bytes):
        return raw_path.decode("utf-8", errors="ignore")
    return str(raw_path)


def _find_map_kd(mtl_abs):
    if not mtl_abs or not os.path.exists(mtl_abs):
        return "", ""
    try:
        with open(mtl_abs, "r", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.lower().startswith("map_kd "):
                    raw_tex = s.split(None, 1)[1].strip()
                    tex_abs = os.path.normpath(os.path.join(os.path.dirname(mtl_abs), raw_tex))
                    return raw_tex, tex_abs
    except OSError:
        return "", ""
    return "", ""


def debug_texture_binding(pb_module, body_id, source_path=None, component="TextureAudit"):
    """Optional per-body texture diagnostics.

    Enabled only when `A2_TEXTURE_DEBUG=1`.
    """
    if not texture_debug_enabled():
        return

    try:
        visual_data = pb_module.getVisualShapeData(body_id)
    except Exception as exc:
        debug_log(component, "texture debug failed: getVisualShapeData", payload={"body_id": body_id, "error": str(exc)})
        return

    if not visual_data:
        debug_log(component, "texture debug: no visual data", payload={"body_id": body_id, "source_path": source_path})
        return

    for idx, shape in enumerate(visual_data):
        mesh_rel = _decode_mesh_path(shape[4] if len(shape) > 4 else "")
        rgba = shape[7] if len(shape) > 7 else None
        payload = {
            "body_id": body_id,
            "source_path": source_path,
            "shape_index": idx,
            "mesh_path": mesh_rel,
            "rgba": rgba,
        }

        if not mesh_rel or not mesh_rel.lower().endswith(".obj"):
            payload["status"] = "skip_non_obj"
            debug_log(component, "texture binding", payload=payload)
            continue

        mesh_abs = mesh_rel if os.path.isabs(mesh_rel) else os.path.normpath(os.path.join(os.getcwd(), mesh_rel))
        payload["mesh_exists"] = os.path.exists(mesh_abs)
        if not os.path.exists(mesh_abs):
            payload["status"] = "mesh_missing"
            debug_log(component, "texture binding", payload=payload)
            continue

        mtl_abs = ""
        try:
            with open(mesh_abs, "r", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if s.lower().startswith("mtllib "):
                        mtl_name = s.split(None, 1)[1].strip()
                        mtl_abs = os.path.normpath(os.path.join(os.path.dirname(mesh_abs), mtl_name))
                        break
        except OSError as exc:
            payload["status"] = "mesh_read_error"
            payload["error"] = str(exc)
            debug_log(component, "texture binding", payload=payload)
            continue

        payload["mtl_path"] = mtl_abs
        payload["mtl_exists"] = bool(mtl_abs) and os.path.exists(mtl_abs)
        if not payload["mtl_exists"]:
            payload["status"] = "mtl_missing"
            debug_log(component, "texture binding", payload=payload)
            continue

        map_kd, tex_abs = _find_map_kd(mtl_abs)
        payload["map_kd"] = map_kd
        payload["texture_path"] = tex_abs
        payload["texture_exists"] = bool(tex_abs) and os.path.exists(tex_abs)

        if not map_kd:
            # Helpful heuristic: if texture-like files exist but map_Kd is absent, this is usually a material bug.
            tex_candidates = sorted(glob.glob(os.path.join(os.path.dirname(mesh_abs), "*.png")) + glob.glob(os.path.join(os.path.dirname(mesh_abs), "*.jpg")))
            payload["texture_candidates"] = [os.path.basename(x) for x in tex_candidates[:6]]
            payload["status"] = "missing_map_kd"
        elif not payload["texture_exists"]:
            payload["status"] = "map_kd_target_missing"
        else:
            payload["status"] = "ok"
        debug_log(component, "texture binding", payload=payload)


def in_workspace(pos, workspace_limits):
    if pos is None or workspace_limits is None:
        return None
    return (
        workspace_limits[0][0] <= pos[0] <= workspace_limits[0][1]
        and workspace_limits[1][0] <= pos[1] <= workspace_limits[1][1]
    )


def snapshot_env(env, stage, workspace_limits=None, component="ENV", extra=None, level=1):
    if not debug_enabled(level):
        return

    rigid_ids = list(env.obj_ids.get("rigid", []))
    objects = []
    for obj_id in rigid_ids:
        pos = None
        rot = None
        dim = None
        try:
            pos, rot, dim = env.obj_info(obj_id)
        except Exception as exc:
            objects.append({"id": obj_id, "error": str(exc)})
            continue

        labels = []
        if hasattr(env, "obj_labels") and isinstance(env.obj_labels, dict):
            labels = env.obj_labels.get(obj_id, [])
        obj_dir = None
        if hasattr(env, "obj_dirs") and isinstance(env.obj_dirs, dict):
            obj_dir = env.obj_dirs.get(obj_id)

        inside = in_workspace(pos, workspace_limits)
        objects.append(
            {
                "id": obj_id,
                "pos": [round(float(v), 4) for v in pos],
                "label": labels,
                "dir": obj_dir,
                "is_target": obj_id in getattr(env, "target_obj_ids", []),
                "is_reference": obj_id in getattr(env, "reference_obj_ids", []),
                "in_workspace": None if inside is None else bool(inside),
            }
        )

    payload = {
        "stage": stage,
        "num_rigid": len(rigid_ids),
        "target_ids": list(getattr(env, "target_obj_ids", [])),
        "reference_ids": list(getattr(env, "reference_obj_ids", [])),
        "objects": objects,
    }
    if extra is not None:
        payload["extra"] = extra
    debug_log(component, "scene snapshot", payload=payload, level=level)
