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


def main() -> int:
    parser = argparse.ArgumentParser(description="IsaacLab-launched smoke run that queries OpenPI policy server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the cube and move it slightly")
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--apply-to-cube", action="store_true", default=True)
    parser.add_argument("--no-apply-to-cube", dest="apply_to_cube", action="store_false")
    parser.add_argument("--delta-scale", type=float, default=0.02)
    parser.add_argument("--disable-warp-cuda", action="store_true", default=False)
    parser.add_argument("--dummy-image", action="store_true", default=False)
    parser.add_argument("--log-file", default="")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    if args_cli.disable_warp_cuda:
        os.environ.setdefault("WARP_DISABLE_CUDA", "1")

    log_fp = None
    if args_cli.log_file:
        log_path = pathlib.Path(args_cli.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("a", encoding="utf-8")

    def _log(msg: str) -> None:
        print(msg, flush=True)
        if log_fp is not None:
            log_fp.write(msg + "\n")
            log_fp.flush()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    openpi_root = _resolve_openpi_root()
    _add_openpi_to_syspath(openpi_root)
    from openpi_client import image_tools

    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext

    _log(f"connecting policy ws://{args_cli.host}:{args_cli.port}")
    client = _create_client(args_cli.host, args_cli.port)
    _log("connected policy")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/light", light_cfg)

    cube_cfg = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.08)),
    )
    cube = RigidObject(cfg=cube_cfg)

    camera = None
    if not args_cli.dummy_image:
        from isaaclab.sensors.camera import Camera, CameraCfg

        camera_cfg = CameraCfg(
            prim_path="/World/CameraSensor",
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
        )
        camera = Camera(cfg=camera_cfg)

    sim.reset()
    cube.reset()
    sim_dt = sim.get_physics_dt()

    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor([[0.8, 0.0, 0.65]], device=sim.device),
            torch.tensor([[0.35, 0.0, 0.08]], device=sim.device),
        )

    for _ in range(60):
        cube.write_data_to_sim()
        sim.step()
        cube.update(sim_dt)
        if camera is not None:
            camera.update(dt=sim_dt)
    _log("sim warmup done")

    state = np.zeros((8,), dtype=np.float32)

    for step in range(int(args_cli.steps)):
        cube.write_data_to_sim()
        sim.step()
        cube.update(sim_dt)
        if args_cli.dummy_image:
            rgb_p = np.zeros((int(args_cli.resize_size), int(args_cli.resize_size), 3), dtype=np.uint8)
        else:
            assert camera is not None
            camera.update(dt=sim_dt)
            rgb_t = camera.data.output["rgb"][0]
            rgb = rgb_t.detach().cpu().numpy()
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb = np.ascontiguousarray(rgb)
            rgb_p = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rgb, args_cli.resize_size, args_cli.resize_size)
            )

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

        first = np.asarray(actions[0], dtype=np.float32).reshape(-1)
        if step % 10 == 0:
            _log(
                f"step={step} action0="
                f"{np.array2string(first, precision=4, separator=',', suppress_small=True)}",
            )

        if args_cli.apply_to_cube and first.size >= 3:
            root_state = cube.data.root_state_w.clone()
            delta = torch.tensor(first[:3], device=sim.device, dtype=torch.float32).view(1, 3) * float(
                args_cli.delta_scale
            )
            root_state[:, :3] = root_state[:, :3] + delta
            root_state[:, 2] = torch.clamp(root_state[:, 2], min=0.02)
            cube.write_root_pose_to_sim(root_state[:, :7])

    simulation_app.close()
    if log_fp is not None:
        log_fp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
