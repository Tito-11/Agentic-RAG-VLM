import os


def main() -> int:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    simulation_app.close()
    print("Isaac Sim headless OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
