from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np


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
            conn = websockets.sync.client.connect(
                uri,
                compression=None,
                max_size=None,
                ping_interval=120,
                ping_timeout=120,
                close_timeout=30,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim smoke run that queries OpenPI policy server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the cube and move it slightly")
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--gui", dest="headless", action="store_false")
    parser.add_argument("--apply-to-cube", action="store_true", default=True)
    parser.add_argument("--no-apply-to-cube", dest="apply_to_cube", action="store_false")
    parser.add_argument("--delta-scale", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    openpi_root = _resolve_openpi_root()
    _add_openpi_to_syspath(openpi_root)
    from openpi_client import image_tools

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        from omni.isaac.core import World
        from omni.isaac.core.objects import DynamicCuboid
        from omni.isaac.sensor import Camera

        client = _create_client(args.host, args.port)

        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
        cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([0.35, 0.0, 0.05], dtype=np.float32),
                scale=np.array([0.06, 0.06, 0.06], dtype=np.float32),
                color=np.array([0.9, 0.1, 0.1], dtype=np.float32),
            )
        )
        camera = Camera(
            prim_path="/World/Camera",
            position=np.array([0.65, 0.0, 0.55], dtype=np.float32),
            frequency=30,
            resolution=(args.resize_size, args.resize_size),
        )
        camera.look_at(np.array([0.35, 0.0, 0.05], dtype=np.float32))
        camera.initialize()
        world.reset()

        for _ in range(30):
            world.step(render=True)

        state = np.zeros((8,), dtype=np.float32)

        for step in range(args.steps):
            world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is None:
                continue
            rgb = rgba[..., :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            rgb = np.ascontiguousarray(rgb)
            rgb_p = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, args.resize_size, args.resize_size))

            payload = {
                "observation/image": rgb_p,
                "observation/state": state,
                "prompt": str(args.prompt),
            }
            inference = client.infer(payload)
            actions = inference.get("actions")
            if not actions:
                raise RuntimeError(f"Malformed response: {inference}")

            first = np.asarray(actions[0], dtype=np.float32).reshape(-1)
            if step % 10 == 0:
                print(
                    f"step={step} action0="
                    f"{np.array2string(first, precision=4, separator=',', suppress_small=True)}"
                )

            if args.apply_to_cube and first.size >= 3:
                pos, _ = cube.get_world_pose()
                delta = first[:3] * float(args.delta_scale)
                new_pos = np.asarray(pos, dtype=np.float32) + delta
                new_pos[2] = max(float(new_pos[2]), 0.02)
                cube.set_world_pose(position=new_pos)

        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())

