#!/usr/bin/env python
"""
Real-camera bronchoscopy navigator — minimal capture script.
Replaces Pyrender simulation with a live camera feed (webcam / USB endoscope).

Workflow:
  b → 到达分叉口 (trigger bifurcation detection)
  1/2/3... → 选择进入第 N 个分支
  e → 到达尽头 (trigger DFS backtrack)
  r → 手动回溯
  t → 显示探索树
  q → 退出

Dependencies: navigator.py + checkpoints_LD/lumen_detection_model.pth + Airways/Network_siliconmodel3.obj
"""

import os, sys

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR  # assume we're in the repo root
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import cv2
import numpy as np
import torch

from navigator import BronchoscopeNavigator

# ── Config ──────────────────────────────────────────────────────────────────
CAMERA_ID = 1                 # 0 = default webcam; change if using USB endoscope
AIRWAY_OBJ = "Airways/Network_siliconmodel3.obj"
LUMEN_MODEL = "checkpoints_LD/lumen_detection_model.pth"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ── Init ────────────────────────────────────────────────────────────────────
print("Loading navigator...")
nav = BronchoscopeNavigator(
    airway_obj_path=AIRWAY_OBJ,
    lumen_model_path=LUMEN_MODEL,
)
print("Navigator ready.")

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print(f"ERROR: Cannot open camera {CAMERA_ID}")
    sys.exit(1)

print(f"Camera {CAMERA_ID} opened ({FRAME_WIDTH}x{FRAME_HEIGHT})")
print()
print("Controls:")
print("  [b] Bifurcation reached — detect branch openings")
print("  [1-9] Select branch (1 = first opening, 2 = second, ...)")
print("  [e] End reached — DFS backtrack to next unexplored branch")
print("  [r] Manual backtrack one level")
print("  [t] Draw exploration tree")
print("  [s] Show status")
print("  [q] Quit")
print()

# ── State ───────────────────────────────────────────────────────────────────
annotated_frame = None   # frame with branch markers overlaid
show_mode = "live"       # "live" or "annotated"

# ── Main loop ───────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    # Mirror for natural feel
    frame = cv2.flip(frame, 1)

    # Decide what to show
    if show_mode == "annotated" and annotated_frame is not None:
        display = annotated_frame
        cv2.putText(display, "[ANNOTATED] Press any key to return", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        display = frame
        cv2.putText(display, "[LIVE] b=bifur | 1-9=branch | e=end | r=back | q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Bronchoscope Navigator", display)
    key = cv2.waitKey(1) & 0xFF

    # ── Key handlers ────────────────────────────────────────────────────────
    if key == ord('q'):
        break

    elif key == ord('b'):
        # Trigger bifurcation detection
        print("[KEY] Bifurcation reached — detecting branch openings...")
        try:
            # Navigator expects RGB image (H,W,3) or (3,H,W)
            # If frame is BGR from OpenCV, convert
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated, info = nav.trigger_bifurcation_reached(rgb_frame)
            # annotated is (H,W,3) RGB — convert back to BGR for OpenCV display
            annotated_frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            show_mode = "annotated"
            print(f"  -> Found {info.get('num_branches', '?')} openings")
            print(f"  -> Node: {info.get('node_id', '?')}")
            if 'unexplored' in info:
                print(f"  -> Unexplored: {info['unexplored']}")
            if 'explored' in info:
                print(f"  -> Explored: {info['explored']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            show_mode = "live"

    elif ord('1') <= key <= ord('9'):
        branch_idx = key - ord('1')  # 0-based
        print(f"[KEY] Selecting branch {branch_idx + 1}...")
        try:
            nav.trigger_branch_selected(branch_idx)
            print(f"  -> Entered branch {branch_idx + 1}")
        except Exception as e:
            print(f"  ERROR: {e}")
        show_mode = "live"
        annotated_frame = None

    elif key == ord('e'):
        print("[KEY] End reached — DFS backtrack...")
        try:
            suggestion = nav.trigger_reached_end()
            if suggestion is not None:
                print(f"  -> Next target: node {suggestion.get('id', '?')}")
                print(f"  -> Frontiers remaining: {suggestion.get('frontiers', '?')}")
            else:
                print("  -> All branches explored!")
        except Exception as e:
            print(f"  ERROR: {e}")
        show_mode = "live"
        annotated_frame = None

    elif key == ord('r'):
        print("[KEY] Manual backtrack...")
        try:
            guidance = nav.trigger_backtrack()
            if guidance is not None:
                print(f"  -> Back to node {guidance.get('id', '?')}")
            else:
                print("  -> Already at root, cannot backtrack further.")
        except Exception as e:
            print(f"  ERROR: {e}")
        show_mode = "live"
        annotated_frame = None

    elif key == ord('s'):
        status = nav.get_status()
        print("─" * 40)
        print("  Status:")
        for k, v in status.items():
            print(f"    {k}: {v}")
        print("─" * 40)

    elif key == ord('t'):
        print("[KEY] Drawing exploration tree...")
        try:
            nav.draw_tree("exploration_tree.png")
            print("  -> Saved to exploration_tree.png")
            # Try to show the tree image
            tree_img = cv2.imread("exploration_tree.png")
            if tree_img is not None:
                cv2.imshow("Exploration Tree", tree_img)
        except Exception as e:
            print(f"  ERROR: {e}")

    else:
        # Any other key returns to live view
        show_mode = "live"
        annotated_frame = None

# ── Cleanup ─────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

# Save final tree
try:
    nav.draw_tree("exploration_tree_final.png")
    print("Final tree saved to exploration_tree_final.png")
except:
    pass

print("Done.")
