# PICO live streaming

GMR receives PICO body tracking through XRoboToolkit. The validated setup uses
an Ubuntu x86_64 host, a PICO 4 Ultra, and the separately built
`xrobotoolkit_sdk` Python binding.

## 1. Install the headset client

Download the latest APK from the
[XRoboToolkit Unity Client releases](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases)
and install it on the headset. The release page also provides the Unity package
for custom client builds.

Enable full-body tracking in the XRoboToolkit application. The PICO and the host
running GMR must be on the same network.

## 2. Install the PC service

Download the Ubuntu package matching the host from the
[XRoboToolkit PC Service releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases).
Packages are available for Ubuntu 22.04, Ubuntu 24.04, and ARM64.

Install the downloaded package:

```bash
sudo apt install ./XRoboToolkit_PC_Service_*.deb
```

Start the service:

```bash
/opt/apps/roboticsservice/runService.sh
```

Open XRoboToolkit on the headset, select the host running the service, enable
full-body tracking, and turn on data sending. The upstream
[client guide](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client#pose-sync-between-xr-device-and-robot-pc)
documents the headset controls.

## 3. Build the Python binding

The binding is not published on PyPI. From the GMR repository, run:

```bash
./scripts/install_pico_binding.sh
```

The script fetches pinned revisions of the PC Service SDK and its Python
binding, builds them in a temporary directory, installs the result into GMR's
uv environment, and verifies the import. It requires Linux, Git, uv, and a
working C++ build toolchain. The PC Service package from the previous step must
also remain installed because it supplies the runtime service and libraries.

See the
[binding repository](https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind)
for Windows build instructions.

## 4. Stream into GMR

Start the PC service and headset data sending before launching GMR:

```bash
uv run stream --source pico --robot unitree_g1
```

Use `--fps` to override the nominal 60 Hz source rate. PICO currently has a
validated built-in profile for `unitree_g1`.

If importing `xrobotoolkit_sdk` fails, reinstall the binding into the same
environment used to run GMR. If no frames arrive, confirm that the PC service
is running, both devices are on the same network, and full-body data sending is
enabled in the headset client.
