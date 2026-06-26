"""Headless test - runs directly (no subprocess) to keep OpenGL context."""
import os, sys

# Find and cd to the github directory
base = r"C:\Users\zhihao\Documents\leizhihao_lit\LIu_science"
for entry in os.listdir(base):
    full = os.path.join(base, entry)
    if os.path.isdir(full) and ("github" in entry.lower() or "AI-Human" in entry):
        os.chdir(full)
        break
else:
    print("ERROR: github directory not found")
    sys.exit(1)

print("Working dir:", os.getcwd())
sys.path.insert(0, os.getcwd())

# Must set before any imports that need OpenGL/Matplotlib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use("Agg")

# Monkey-patch PyOpenGL glGenTextures to fix ctypes issue on Windows + Python 3.12
import OpenGL.GL as _gl
import ctypes
from OpenGL.raw.GL.VERSION.GL_1_1 import glGenTextures as _raw_glGenTextures
def _fixed_glGenTextures(n, textures=None):
    if textures is None:
        arr = (ctypes.c_uint * n)()
        _raw_glGenTextures(n, arr)
        return arr[0] if n == 1 else list(arr)
    _raw_glGenTextures(n, textures)
_gl.glGenTextures = _fixed_glGenTextures

# Monkey-patch mayavi show
import mayavi.mlab as mlab
_mlab_show = mlab.show
def _patched_show(*args, **kwargs):
    figure = kwargs.get('figure', None)
    if figure is None and args:
        figure = args[0]
    try:
        os.makedirs("results", exist_ok=True)
        mlab.savefig("results/mayavi_output.png", figure=figure, magnification=2)
        print("[patched] Saved figure to results/mayavi_output.png")
    except Exception as e:
        print(f"[patched] savefig failed: {e}")
mlab.show = _patched_show

# Patch matplotlib show
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None

# Now the actual test
import torch
import torchvision

from lib.network.model import SingleLevelCILMultiFrame, targetsDetectionNet
from lib.engine.onlineSimulation import onlineSimulationWithNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load policy model
net = SingleLevelCILMultiFrame(10)
ckpt = torch.load("checkpoints_policy/policy_model.pth", map_location=device)
md = net.state_dict()
ckpt = {k: v for k, v in ckpt.items() if k in md}
md.update(ckpt)
net.load_state_dict(md)
net.to(device=device)
print("Policy model loaded.")

# Load lumen detection model
td_net = targetsDetectionNet()
ckpt = torch.load("checkpoints_LD/lumen_detection_model.pth", map_location=device)
md = td_net.state_dict()
ckpt = {k: v for k, v in ckpt.items() if k in md}
md.update(ckpt)
td_net.load_state_dict(md)
td_net.to(device=device)
print("Lumen detection model loaded.")

# Run simulation
from argparse import Namespace
args = Namespace(history_length=10)

transform_eval = torchvision.transforms.ToTensor()
with torch.no_grad():
    net.eval()
    td_net.eval()
    sim = onlineSimulationWithNetwork(args, 'siliconmodel3 Centerline model', renderer='pyrender', training=False)
    sim.run(net, td_net, epoch=None, transform_func=transform_eval, training=False)

print("\n=== TEST COMPLETE ===")
