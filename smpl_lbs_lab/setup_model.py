"""Locate or prepare SMPL_NEUTRAL.pkl for smplx."""

from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy import sparse

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "smpl"
TARGET = MODEL_DIR / "SMPL_NEUTRAL.pkl"

# Standard SMPL kinematic tree (24 joints)
KINTREE_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]


def convert_py2_pickle(src: Path, dst: Path) -> None:
    """Convert legacy Python 2 SMPL pickle to Python 3."""
    with open(src, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _candidate_paths() -> list[Path]:
    names = [
        "SMPL_NEUTRAL.pkl",
        "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
        "basicModel_neutral_lbs_10_207_0_v1.1.0.pkl",
        "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
    ]
    dirs = [
        MODEL_DIR,
        ROOT / "models",
        ROOT,
        Path.home() / "Downloads",
        Path.home() / ".cache" / "phalp" / "3D" / "models" / "smpl",
        Path.home() / ".cache" / "4DHumans" / "data" / "smpl",
    ]
    paths: list[Path] = []
    for d in dirs:
        for n in names:
            paths.append(d / n)
    return paths


def _make_tpose_humanoid() -> trimesh.Trimesh:
  """Build a coarse T-pose body for fallback visualization."""
  parts: list[trimesh.Trimesh] = []

  def cyl(r, h, transform=None):
      m = trimesh.creation.cylinder(radius=r, height=h, sections=24)
      if transform is not None:
          m.apply_transform(transform)
      return m

  # Torso
  parts.append(cyl(0.14, 0.42).apply_translation([0, 0, 0.22]))
  parts.append(cyl(0.12, 0.18).apply_translation([0, 0, 0.58]))
  # Head
  parts.append(trimesh.creation.icosphere(subdivisions=3, radius=0.10).apply_translation([0, 0, 0.78]))
  # Arms
  rot_y = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
  for sx in (-1, 1):
      shoulder = np.array([0.22 * sx, 0, 0.62])
      parts.append(cyl(0.045, 0.28, rot_y).apply_translation(shoulder + np.array([0.14 * sx, 0, 0])))
      parts.append(cyl(0.04, 0.26, rot_y).apply_translation(shoulder + np.array([0.40 * sx, 0, -0.02])))
      parts.append(cyl(0.035, 0.18, rot_y).apply_translation(shoulder + np.array([0.60 * sx, 0, -0.04])))
  # Legs
  for sx in (-1, 1):
      parts.append(cyl(0.055, 0.42).apply_translation([0.09 * sx, 0, -0.05]))
      parts.append(cyl(0.045, 0.40).apply_translation([0.09 * sx, 0, -0.47]))
      parts.append(cyl(0.04, 0.12).apply_translation([0.09 * sx, 0.06, -0.78]))

  mesh = trimesh.util.concatenate(parts)
  mesh.merge_vertices()
  while len(mesh.vertices) < 6890:
      mesh = mesh.subdivide()
  if len(mesh.vertices) > 6890:
      idx = np.linspace(0, len(mesh.vertices) - 1, 6890).astype(np.int64)
      v = mesh.vertices[idx]
      # keep first 13776 faces if possible
      f = mesh.faces
      if len(f) > 13776:
          f = f[:13776]
      elif len(f) < 13776:
          pad = np.tile(f[:1], (13776 - len(f), 1))
          f = np.vstack([f, pad])
      mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
  return mesh


def _joint_positions_template(v: np.ndarray) -> np.ndarray:
    """Heuristic rest joint positions for fallback model."""
    joints = np.zeros((24, 3), dtype=np.float64)
    joints[0] = [0, 0, 0.35]
    joints[1] = [-0.09, 0, 0.30]
    joints[2] = [0.09, 0, 0.30]
    joints[3] = [0, 0, 0.48]
    joints[4] = [-0.09, 0, 0.02]
    joints[5] = [0.09, 0, 0.02]
    joints[6] = [0, 0, 0.58]
    joints[7] = [-0.09, 0, -0.38]
    joints[8] = [0.09, 0, -0.38]
    joints[9] = [0, 0, 0.66]
    joints[10] = [-0.09, 0.06, -0.78]
    joints[11] = [0.09, 0.06, -0.78]
    joints[12] = [0, 0, 0.72]
    joints[13] = [-0.05, 0, 0.68]
    joints[14] = [0.05, 0, 0.68]
    joints[15] = [0, 0, 0.82]
    joints[16] = [-0.22, 0, 0.62]
    joints[17] = [0.22, 0, 0.62]
    joints[18] = [-0.50, 0, 0.58]
    joints[19] = [0.50, 0, 0.58]
    joints[20] = [-0.70, 0, 0.54]
    joints[21] = [0.70, 0, 0.54]
    joints[22] = [-0.82, 0, 0.52]
    joints[23] = [0.82, 0, 0.52]
    return joints


def _compute_lbs_weights(v: np.ndarray, joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Inverse-distance soft weights over kinematic bones."""
    n_verts = v.shape[0]
    n_joints = joints.shape[0]
    w = np.zeros((n_verts, n_joints), dtype=np.float64)

    for j in range(n_joints):
        if j == 0:
            bone_pts = joints[[0, 3]]
        else:
            bone_pts = joints[[parents[j], j]]
        seg = bone_pts[1] - bone_pts[0]
        seg_len = np.linalg.norm(seg) + 1e-8
        t = np.clip(((v - bone_pts[0]) @ seg) / (seg_len ** 2), 0.0, 1.0)
        proj = bone_pts[0] + t[:, None] * seg
        dist = np.linalg.norm(v - proj, axis=1)
        w[:, j] = 1.0 / (dist ** 2 + 1e-4)

    w /= w.sum(axis=1, keepdims=True)
    return w


def _build_j_regressor(v: np.ndarray, joints: np.ndarray) -> sparse.csc_matrix:
    """One-hot regressor from nearest template vertex per joint."""
    rows, cols, data = [], [], []
    for j in range(joints.shape[0]):
        idx = int(np.argmin(np.linalg.norm(v - joints[j], axis=1)))
        rows.append(j)
        cols.append(idx)
        data.append(1.0)
    return sparse.csc_matrix((data, (rows, cols)), shape=(24, v.shape[0]))


def generate_fallback_model(dst: Path = TARGET) -> Path:
    """Create SMPL-compatible fallback pickle when official model is absent."""
    mesh = _make_tpose_humanoid()
    v_template = mesh.vertices.astype(np.float64)
    faces = mesh.faces.astype(np.int64)
    if len(faces) < 13776:
        pad = np.tile(faces[:1], (13776 - len(faces), 1))
        faces = np.vstack([faces, pad])
    else:
        faces = faces[:13776]

    joints = _joint_positions_template(v_template)
    weights = _compute_lbs_weights(v_template, joints, KINTREE_PARENTS)
    j_reg = _build_j_regressor(v_template, joints)

    shapedirs = np.random.RandomState(0).randn(6890, 3, 10).astype(np.float64) * 0.002
    shapedirs[:, 2, 0] += np.linspace(-0.03, 0.03, 6890)
    posedirs = np.random.RandomState(1).randn(6890, 3, 207).astype(np.float64) * 0.001

    kintree_table = np.zeros((2, 24), dtype=np.int64)
    kintree_table[0] = np.arange(24)
    kintree_table[1] = KINTREE_PARENTS

    payload = {
        "v_template": v_template,
        "f": faces,
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "J_regressor": j_reg,
        "weights": weights,
        "kintree_table": kintree_table,
        "J": joints,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dst


def ensure_model(force_fallback: bool = False) -> tuple[Path, bool]:
    """
    Return (model_path, is_official).
    Copies/converts discovered pickles into models/smpl/SMPL_NEUTRAL.pkl.
    """
    if TARGET.exists() and not force_fallback:
        return TARGET, not (ROOT / ".using_fallback_model").exists()

    if not force_fallback:
        for p in _candidate_paths():
            if p.exists() and p.resolve() != TARGET.resolve():
                if p.name != "SMPL_NEUTRAL.pkl":
                    from scripts.convert_smpl_pkl import convert_smpl_pkl
                    convert_smpl_pkl(p, TARGET)
                else:
                    try:
                        import smplx  # noqa: F401
                        shutil.copy2(p, TARGET)
                    except Exception:
                        from scripts.convert_smpl_pkl import convert_smpl_pkl
                        convert_smpl_pkl(p, TARGET)
                (ROOT / ".using_fallback_model").unlink(missing_ok=True)
                return TARGET, True

    generate_fallback_model(TARGET)
    (ROOT / ".using_fallback_model").write_text("1", encoding="utf-8")
    return TARGET, False


if __name__ == "__main__":
    path, official = ensure_model()
    flag = "official/converted" if official else "fallback procedural"
    print(f"Model ready: {path} [{flag}]")
