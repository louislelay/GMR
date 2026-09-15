# Xsens MVN live streaming

GMR receives Position + Orientation quaternion datagrams from Xsens MVN through
the optional `xsens_mvn_robot` parser. The parser currently publishes Linux
x86_64 and ARM64 wheels for CPython 3.10 through 3.13.

## 1. Install the parser

From the GMR repository, run:

```bash
./scripts/install_xsens_parser.sh
```

The script detects GMR's uv-managed Python version and the host architecture,
selects the matching pinned wheel, installs it into the project environment,
and verifies the import. The
[parser repository](https://github.com/jiminghe/xsens_mvn_robot_python)
currently provides wheels for CPython 3.10 through 3.13 on Linux x86_64 and
ARM64. It does not currently provide macOS or Windows wheels.

## 2. Configure MVN

Open Xsens MVN and configure its Network Streamer:

1. Open **Options → Network Streamer**.
2. Add a stream destination.
3. Set the destination to `127.0.0.1` when GMR runs on the same machine, or to
   the GMR host's LAN address when it runs on another machine.
4. Set the UDP port to `9763`, or choose another port and pass the same value to
   GMR.
5. Enable **Position + Orientation (Quaternion)**. GMR does not require the
   other stream payloads.
6. Start a live MVN session or replay a recorded MVN session and confirm that
   the Network Streamer is active.

For cross-machine streaming, both machines must be on the same reachable
network and the receiver firewall must allow the selected UDP port.

## 3. Stream into GMR

MVN must already be sending data when GMR starts because parser initialization
waits for the first quaternion datagram:

```bash
uv run stream \
  --source xsens \
  --robot unitree_g1 \
  --port 9763 \
  --human-height 1.8
```

`--human-height` is optional but recommended when the performer's measured
height is known. Xsens currently has a validated built-in profile for
`unitree_g1`.

If initialization fails, verify the destination address, UDP port, streamer
status, and firewall. If GMR reports missing bodies, confirm that MVN is
streaming a full-body Position + Orientation quaternion payload.
