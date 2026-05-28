from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _resolve_openpi_root() -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if env_root := os.environ.get("AGENTIC_VLA_OPENPI_ROOT"):
        candidates.append(pathlib.Path(env_root).expanduser())
    candidates.append(pathlib.Path("/home/admin1/ct/openpi-official"))
    candidates.append(PROJECT_ROOT / "openpi")
    for candidate in candidates:
        if (candidate / "packages" / "openpi-client" / "src" / "openpi_client").exists():
            return candidate.resolve()
    raise RuntimeError(
        "Unable to locate openpi checkout with openpi-client. "
        "Set AGENTIC_VLA_OPENPI_ROOT to the openpi-official checkout."
    )


def _add_openpi_to_syspath(openpi_root: pathlib.Path) -> None:
    sys.path.insert(0, str(openpi_root / "packages" / "openpi-client" / "src"))
    if (openpi_root / "src" / "openpi").exists():
        sys.path.insert(0, str(openpi_root / "src"))


def _create_client(host: str, port: int, max_attempts: int = 10):
    import websockets.sync.client
    from openpi_client import msgpack_numpy as _msgpack_numpy
    from openpi_client import websocket_client_policy as _ws_policy

    uri = f"ws://{host}:{port}"
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            try:
                conn = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    ping_interval=120,
                    ping_timeout=120,
                    close_timeout=30,
                )
            except TypeError:
                conn = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                )
            metadata = _msgpack_numpy.unpackb(conn.recv())
            client = _ws_policy.WebsocketClientPolicy.__new__(_ws_policy.WebsocketClientPolicy)
            client._uri = uri
            client._api_key = None
            client._packer = _msgpack_numpy.Packer()
            client._ws = conn
            client._server_metadata = metadata
            return client
        except Exception as exc:
            last_err = exc
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to connect to policy server at {uri}: {last_err}")


def _quat_wxyz_to_axis_angle(q) -> "tuple[float, float, float]":
    import math

    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-9:
        return (0.0, 0.0, 0.0)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    qw = max(-1.0, min(1.0, qw))
    angle = 2.0 * math.acos(qw)
    s = math.sqrt(max(1e-12, 1.0 - qw * qw))
    axis = (qx / s, qy / s, qz / s)
    return (axis[0] * angle, axis[1] * angle, axis[2] * angle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Franka Panda in IsaacLab and query OpenPI policy server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--arm-scale", type=float, default=0.05)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.0)
    parser.add_argument("--disable-warp-cuda", action="store_true", default=False)
    parser.add_argument("--dummy-image", action="store_true", default=False)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    if args_cli.disable_warp_cuda:
        os.environ.setdefault("WARP_DISABLE_CUDA", "1")

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    openpi_root = _resolve_openpi_root()
    _add_openpi_to_syspath(openpi_root)
    from openpi_client import image_tools

    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    from isaaclab.assets import Articulation
    from isaaclab.controllers.differential_ik import DifferentialIKController
    from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

    print(f"connecting policy ws://{args_cli.host}:{args_cli.port}", flush=True)
    client = _create_client(args_cli.host, args_cli.port)
    print("connected policy", flush=True)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 2.2], [0.55, 0.0, 1.0])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)).func("/World/light", sim_utils.DomeLightCfg())
    sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd").func(
        "/World/Table",
        sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        translation=(0.55, 0.0, 0.0),
        orientation=(0.70711, 0.0, 0.0, 0.70711),
    )

    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 0.0)
    robot = Articulation(cfg=robot_cfg)

    camera = None
    if not args_cli.dummy_image:
        from isaaclab.sensors.camera import Camera, CameraCfg

        camera_cfg = CameraCfg(
            prim_path="/World/Camera",
            update_period=0,
            height=int(args_cli.resize_size),
            width=int(args_cli.resize_size),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(pos=(-0.6, 0.0, 1.6), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        )
        camera = Camera(camera_cfg)

    ee_body_ids, ee_body_names = robot.find_bodies("panda_hand")
    if len(ee_body_ids) != 1:
        raise RuntimeError(f"Failed to resolve ee body 'panda_hand': {ee_body_names}")
    ee_body_idx = int(ee_body_ids[0])

    arm_joint_ids, _ = robot.find_joints(["panda_joint.*"])
    if len(arm_joint_ids) == 0:
        raise RuntimeError("Failed to resolve arm joints: panda_joint.*")
    arm_joint_ids = list(map(int, arm_joint_ids))

    gripper_joint_ids, _ = robot.find_joints(["panda_finger_joint.*"])
    gripper_joint_ids = list(map(int, gripper_joint_ids))

    if robot.is_fixed_base:
        jacobi_body_idx = ee_body_idx - 1
        jacobi_joint_ids = arm_joint_ids
    else:
        jacobi_body_idx = ee_body_idx
        jacobi_joint_ids = [i + 6 for i in arm_joint_ids]

    controller = DifferentialIKController(
        cfg=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        num_envs=1,
        device=sim.device,
    )

    sim.reset()
    for _ in range(60):
        sim.step()
        robot.update(sim.get_physics_dt())
    print("sim warmup done", flush=True)

    action_queue: list[np.ndarray] = []

    for step in range(int(args_cli.steps)):
        if not simulation_app.is_running():
            break

        sim.step()
        sim_dt = sim.get_physics_dt()
        robot.update(sim_dt)

        ee_pos_w = robot.data.body_pos_w[0, ee_body_idx].detach().cpu().numpy()
        ee_quat_w = robot.data.body_quat_w[0, ee_body_idx].detach().cpu().numpy()
        gripper_open_fraction = 0.0
        if gripper_joint_ids:
            gl = robot.data.soft_joint_pos_limits[0, gripper_joint_ids].detach().cpu().numpy()
            gp = robot.data.joint_pos[0, gripper_joint_ids].detach().cpu().numpy()
            denom = np.maximum(gl[:, 1] - gl[:, 0], 1e-6)
            frac = (gp - gl[:, 0]) / denom
            gripper_open_fraction = float(np.clip(np.mean(frac), 0.0, 1.0))

        if args_cli.dummy_image:
            rgb = np.zeros((int(args_cli.resize_size), int(args_cli.resize_size), 3), dtype=np.uint8)
        else:
            assert camera is not None
            camera.update(dt=sim_dt)
            rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb = np.ascontiguousarray(rgb)
        rgb_p = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, args_cli.resize_size, args_cli.resize_size))

        state = np.concatenate(
            [
                ee_pos_w.astype(np.float32),
                np.asarray(_quat_wxyz_to_axis_angle(ee_quat_w), dtype=np.float32),
                np.asarray([gripper_open_fraction], dtype=np.float32),
            ],
            axis=0,
        )

        if not action_queue:
            payload = {
                "observation/image": rgb_p,
                "observation/wrist_image": rgb_p,
                "observation/state": state,
                "prompt": str(args_cli.prompt),
            }
            inference = client.infer(payload)
            actions = inference.get("actions")
            if actions is None or len(actions) == 0:
                raise RuntimeError(f"Malformed response: {inference}")
            action_queue = [np.asarray(a, dtype=np.float32).reshape(-1) for a in actions[: int(args_cli.replan_steps)]]

        act = action_queue.pop(0)
        if act.size < 6:
            raise RuntimeError(f"Expected >=6D action, got {act.size}")

        arm_delta = torch.tensor(act[:6], device=sim.device, dtype=torch.float32).view(1, 6) * float(args_cli.arm_scale)

        ee_pos_w_t = robot.data.body_pos_w[:, ee_body_idx]
        ee_quat_w_t = robot.data.body_quat_w[:, ee_body_idx]
        root_pos_w_t = robot.data.root_pos_w
        root_quat_w_t = robot.data.root_quat_w
        ee_pos_b_t, ee_quat_b_t = math_utils.subtract_frame_transforms(root_pos_w_t, root_quat_w_t, ee_pos_w_t, ee_quat_w_t)

        controller.set_command(arm_delta, ee_pos_b_t, ee_quat_b_t)

        jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, jacobi_joint_ids]
        base_rot = robot.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        joint_pos = robot.data.joint_pos[:, arm_joint_ids]
        joint_pos_des = controller.compute(ee_pos_b_t, ee_quat_b_t, jacobian, joint_pos)
        robot.set_joint_position_target(joint_pos_des, arm_joint_ids)

        if gripper_joint_ids and act.size >= 7:
            open_fraction = 1.0 if float(act[6]) >= float(args_cli.gripper_open_threshold) else 0.0
            gl = robot.data.soft_joint_pos_limits[:, gripper_joint_ids]
            target = gl[..., 0] + open_fraction * (gl[..., 1] - gl[..., 0])
            robot.set_joint_position_target(target, gripper_joint_ids)

        robot.write_data_to_sim()

        if step % 20 == 0:
            print(
                f"step={step} state_dim={state.size} action_dim={act.size} "
                f"ee_pos=({ee_pos_w[0]:.3f},{ee_pos_w[1]:.3f},{ee_pos_w[2]:.3f}) gripper={gripper_open_fraction:.2f}"
                ,
                flush=True,
            )

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
