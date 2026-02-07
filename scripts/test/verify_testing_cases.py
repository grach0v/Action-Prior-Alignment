import ast
import os
import re
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONSTANTS_PATH = os.path.join(ROOT, "env", "constants.py")


def extract_list(text, name):
    pattern = rf"^{name}\s*=\s*(\[.*?\])\s*$"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not match:
        return []
    return ast.literal_eval(match.group(1))


def load_label_sets():
    with open(CONSTANTS_PATH, "r") as f:
        text = f.read()

    seen_labels = set(extract_list(text, "LABEL"))
    seen_all = set(extract_list(text, "ALL_LABEL"))
    seen_general = set(extract_list(text, "GENERAL_LABEL"))
    seen_color_shape = set(extract_list(text, "COLOR_SHAPE"))
    seen_function = set(extract_list(text, "FUNCTION"))

    unseen_labels = set(extract_list(text, "UNSEEN_LABEL"))
    unseen_general = set(extract_list(text, "UNSEEN_GENERAL_LABEL"))
    unseen_color_shape = set(extract_list(text, "UNSEEN_COLOR_SHAPE"))
    unseen_function = set(extract_list(text, "UNSEEN_FUNCTION"))

    seen_place = seen_all | seen_general | seen_color_shape | seen_function
    unseen_place = unseen_labels | unseen_general | unseen_color_shape | unseen_function

    return {
        "seen_place": seen_place,
        "unseen_place": unseen_place,
        "seen_pick": seen_labels | seen_general | seen_color_shape | seen_function,
        "unseen_pick": unseen_labels
        | unseen_general
        | unseen_color_shape
        | unseen_function,
    }


def is_asset_line(line):
    return line.startswith("assets/") or line.startswith("assets\\")


def validate_asset_paths(lines, errors, case_path):
    for line in lines:
        if not is_asset_line(line):
            continue
        asset_path = line.split()[0]
        full_path = os.path.join(ROOT, asset_path)
        if not os.path.exists(full_path):
            errors.append(f"Missing asset file: {asset_path} in {case_path}")


def validate_place_case(lines, label_set, errors, case_path):
    if len(lines) < 3:
        errors.append(f"Malformed place case (too few lines): {case_path}")
        return
    label = lines[1]
    if label not in label_set:
        errors.append(f"Unknown place label '{label}' in {case_path}")
    validate_asset_paths(lines[3:], errors, case_path)


def validate_pick_case(lines, errors, case_path):
    if len(lines) < 3:
        errors.append(f"Malformed pick case (too few lines): {case_path}")
        return
    validate_asset_paths(lines[2:], errors, case_path)


def validate_pickplace_case(lines, label_set, errors, case_path):
    if len(lines) < 5:
        errors.append(f"Malformed pickplace case (too few lines): {case_path}")
        return

    # Pick section: instruction + target indices + assets until next non-asset line
    idx = 2
    while idx < len(lines) and is_asset_line(lines[idx]):
        idx += 1
    validate_asset_paths(lines[2:idx], errors, case_path)

    if idx + 2 >= len(lines):
        errors.append(f"Malformed pickplace place section: {case_path}")
        return

    # Place section starts at idx (instruction), then label, then relation
    label = lines[idx + 1]
    if label not in label_set:
        errors.append(f"Unknown place label '{label}' in {case_path}")

    validate_asset_paths(lines[idx + 3 :], errors, case_path)


def collect_case_files(root_dir):
    cases = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith(".txt"):
                cases.append(os.path.join(dirpath, name))
    return cases


def main():
    labels = load_label_sets()

    errors = []
    base = os.path.join(ROOT, "testing_cases")

    for case_path in collect_case_files(base):
        with open(case_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        lower_path = case_path.replace("\\", "/")

        if "/grasp_testing_cases/" in lower_path:
            validate_pick_case(lines, errors, case_path)
        elif "/place_testing_cases/" in lower_path:
            label_set = (
                labels["unseen_place"]
                if "/unseen/" in lower_path
                else labels["seen_place"]
            )
            validate_place_case(lines, label_set, errors, case_path)
        elif "/pp_testing_cases/" in lower_path:
            label_set = (
                labels["unseen_place"]
                if "/unseen/" in lower_path
                else labels["seen_place"]
            )
            validate_pickplace_case(lines, label_set, errors, case_path)

    if errors:
        print("Case validation failed:")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)

    print("All testing cases passed validation.")


if __name__ == "__main__":
    main()
