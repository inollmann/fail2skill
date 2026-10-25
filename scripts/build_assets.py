"""Builds the sim-ready ZED mini assets in ``assets/`` from the custom mount CAD.

    ./al.sh run python scripts/build_assets.py

Inputs (hand made):
    assets/Franka_Hand_ZED_mini_Mount.usdc   mount + a simplified ZED mini, in mm, in the Franka hand CAD frame

Outputs (generated, committed):
    assets/zed_mini.usd                  the camera: body, two lenses, left/right ``Camera`` prims
    assets/panda_hand_zed_mount.usda     mount (without its dummy camera) + zed_mini, in m, in the ``panda_hand`` frame

``zed_mini.usd`` uses the ROS optical convention (+z forward, +x right, +y down) with its origin at the
midpoint between the two lenses on the lens plane. The simplified camera body of the mount CAD is reused
as its visual so that it fits the cradle of the mount exactly.

Where the mount sits on the hand is *not* part of the assets, see ``MOUNT_POS`` / ``MOUNT_ROT`` in
``panda_al/tasks/pick_place_env_cfg.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ASSETS = Path(__file__).resolve().parents[1] / "assets"
MOUNT_CAD = ASSETS / "Franka_Hand_ZED_mini_Mount.usdc"
MM = 0.001

# --- ZED mini (datasheet): 124.5 x 30.5 x 26.5 mm, 63 mm baseline, 90 x 60 deg FOV, 0.1 .. 15 m depth, 63 g
BASELINE = 0.063
MASS = 0.063
FOCAL_LENGTH_MM = 2.8
HORIZONTAL_APERTURE_MM = 2 * FOCAL_LENGTH_MM  # tan(45 deg) = 1 -> 90 deg horizontal FOV
VERTICAL_APERTURE_MM = HORIZONTAL_APERTURE_MM * 9 / 16  # 16:9 -> ~59 deg vertical FOV
DEPTH_RANGE = (0.1, 15.0)
LENS_RADIUS = 0.0095

# --- where the CAD's lenses are (mount CAD coordinates, mm); the lens plane is the top of the barrels
LENS_CENTER_CAD = np.array([-51.0, 0.17, 63.25])
# CAD frame (z along the lens axis, y along the baseline) -> ROS optical frame: x = -y, y = x, z = z
CAD_TO_OPTICAL = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def mesh_world_points(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
    return (xf @ np.c_[pts, np.ones(len(pts))].T).T[:, :3], xf[:3, :3]


def preview_material(stage: Usd.Stage, path: str, color, roughness: float, metallic: float = 0.0):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def add_collision(prim: Usd.Prim):
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")


def build_zed_mini(out: Path):
    cad = Usd.Stage.Open(str(MOUNT_CAD))
    cad_mesh = cad.GetPrimAtPath("/Franka_Hand_ZED_mini_Mount/MeshInstance/Körper1")
    pts_mm, rot = mesh_world_points(cad_mesh)
    pts = (CAD_TO_OPTICAL @ (pts_mm - LENS_CENTER_CAD).T).T * MM
    src = UsdGeom.Mesh(cad_mesh)

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0) if hasattr(UsdPhysics, "SetStageKilogramsPerUnit") else None
    root = UsdGeom.Xform.Define(stage, "/zed_mini")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetDocumentation(
        "Stereolabs ZED mini. ROS optical frame (+z forward, +x right, +y down), origin between the lenses on the "
        "lens plane. Cameras: left_cam / right_cam (USD cameras, 16:9, 90 x ~59 deg FOV), depth range in "
        "zed:depthRange."
    )
    root.GetPrim().CreateAttribute("zed:depthRange", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*DEPTH_RANGE))
    root.GetPrim().CreateAttribute("zed:baseline", Sdf.ValueTypeNames.Float).Set(BASELINE)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(MASS)

    body_mat = preview_material(stage, "/zed_mini/Looks/body", (0.06, 0.06, 0.065), 0.45, 0.2)
    glass_mat = preview_material(stage, "/zed_mini/Looks/lens", (0.0, 0.0, 0.0), 0.05)

    body = UsdGeom.Mesh.Define(stage, "/zed_mini/body")
    body.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    body.CreateFaceVertexCountsAttr(src.GetFaceVertexCountsAttr().Get())
    body.CreateFaceVertexIndicesAttr(src.GetFaceVertexIndicesAttr().Get())
    body.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    body.CreateDoubleSidedAttr(False)
    ext = UsdGeom.Boundable.ComputeExtentFromPlugins(body, 0)
    body.CreateExtentAttr(ext)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(body_mat)
    add_collision(body.GetPrim())

    for side, x in (("left", -BASELINE / 2), ("right", BASELINE / 2)):
        lens = UsdGeom.Cylinder.Define(stage, f"/zed_mini/{side}_lens")
        lens.CreateAxisAttr(UsdGeom.Tokens.z)
        lens.CreateRadiusAttr(LENS_RADIUS * 0.6)
        lens.CreateHeightAttr(0.0004)
        UsdGeom.Xformable(lens).AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.0002))
        UsdShade.MaterialBindingAPI.Apply(lens.GetPrim()).Bind(glass_mat)

        cam = UsdGeom.Camera.Define(stage, f"/zed_mini/{side}_cam")
        cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        cam.CreateFocalLengthAttr(FOCAL_LENGTH_MM)
        cam.CreateHorizontalApertureAttr(HORIZONTAL_APERTURE_MM)
        cam.CreateVerticalApertureAttr(VERTICAL_APERTURE_MM)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 20.0))
        xf = UsdGeom.Xformable(cam)
        xf.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.0))
        # USD cameras look down -z with +y up; the optical frame looks down +z with +y down
        xf.AddOrientOp().Set(Gf.Quatf(0.0, 1.0, 0.0, 0.0))
    stage.GetRootLayer().Save()


def build_mount(out: Path):
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/ZedMount")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetDocumentation(
        "Franka_Hand_ZED_mini_Mount + zed_mini in metres. The frame is the CAD frame of the mount, which is "
        "intended to coincide with panda_hand (z towards the fingers) up to the offset set in the env config."
    )

    mount = UsdGeom.Xform.Define(stage, "/ZedMount/mount")
    mount.GetPrim().GetReferences().AddReference(f"./{MOUNT_CAD.name}", "/Franka_Hand_ZED_mini_Mount")
    UsdGeom.Xformable(mount).AddScaleOp().Set(Gf.Vec3f(MM, MM, MM))
    # the CAD includes a dummy camera: replaced by zed_mini
    stage.OverridePrim("/ZedMount/mount/MeshInstance").SetActive(False)
    for name in ("MeshInstance_1", "MeshInstance_2"):
        add_collision(stage.OverridePrim(f"/ZedMount/mount/{name}/Körper1"))

    zed = UsdGeom.Xform.Define(stage, "/ZedMount/zed_mini")
    zed.GetPrim().GetReferences().AddReference("./zed_mini.usd")
    xf = UsdGeom.Xformable(zed)
    xf.AddTranslateOp().Set(Gf.Vec3d(*(LENS_CENTER_CAD * MM)))
    # optical frame in CAD frame: x = -y_cad, y = x_cad, z = z_cad  (-90 deg about z)
    xf.AddOrientOp().Set(Gf.Quatf(float(np.cos(-np.pi / 4)), 0.0, 0.0, float(np.sin(-np.pi / 4))))
    stage.GetRootLayer().Save()


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    for path in (ASSETS / "zed_mini.usd", ASSETS / "panda_hand_zed_mount.usda"):
        path.unlink(missing_ok=True)
    build_zed_mini(ASSETS / "zed_mini.usd")
    build_mount(ASSETS / "panda_hand_zed_mount.usda")
    print(f"[assets] wrote zed_mini.usd and panda_hand_zed_mount.usda in {ASSETS}")


if __name__ == "__main__":
    main()
