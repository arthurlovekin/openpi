"""Measure the one-way command latency from the inference server to the robot computer.

`examples/umi_rizon10/main.py` runs on the robot machine and is the websocket *client*;
`scripts/serve_policy.py` runs on the H100 and is the websocket *server*. An action chunk therefore
travels H100 -> robot on the response leg of `WebsocketClientPolicy.infer()`, and that leg is what
this script measures.

Nothing in the openpi protocol exchanges wall-clock timestamps -- `server_timing.infer_ms` is a
purely local duration -- so a client-only measurement can only produce a round trip. Halving the
round trip is specifically wrong here: the uplink observation is ~150 kB while the downlink action
chunk is ~1 kB, a ~150:1 asymmetry. So this script ships both ends of a dedicated probe:

  * `serve` runs on the H100, alongside (not instead of) the real policy server.
  * `probe` runs on the robot machine and does all of the measuring and reporting.

The clock offset between the two machines is estimated NTP-style from a separate stream of *tiny,
symmetric* exchanges, where "uplink == downlink" is as defensible as it gets, and that offset is
then applied to the realistic asymmetric exchange. That is the only way to separate the two
directions, and it is not exact -- the report prints the resulting error bar next to every one-way
number.

Both ends mirror the production transport deliberately: the same `websockets` entry points, the same
`compression=None, max_size=None`, the same `msgpack_numpy` packer, and the same payload shapes that
`_observe()` in `main.py` sends, so the bytes on the wire match what the deployed system sends.

The probe prints one number -- the H100 -> robot one-way latency, ready to paste into
`downlink_lag_s` in `calibration/*.yaml` -- plus a histogram of the raw samples and any warning
that would make that number untrustworthy. Everything else (all ten derived metrics, their full
percentile tables, and every raw timestamp) goes to `--out` as JSON.

Usage:

    # on the H100 -- port 8001 so it coexists with serve_policy.py on 8000
    uv run examples/umi_rizon10/calibration/measure_command_latency.py serve --port 8001

    # on the robot machine (the flexiv_zed env; needs only openpi-client, numpy, tyro)
    PYTHONPATH=/path/to/openpi/src \
        python examples/umi_rizon10/calibration/measure_command_latency.py \
        probe --host <h100-ip> --port 8001 --samples 200
"""

import asyncio
import dataclasses
import importlib.util
import json
import logging
import pathlib
import sys
import time
import types

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy
import tyro
import websockets
import websockets.asyncio.server
import websockets.sync.client

logger = logging.getLogger(__name__)

# `main.py --open_loop_horizon` default: 8 of a 16-step chunk, so the client re-queries the server
# every 8 control steps. FPS / this is the real request rate and hence the default pacing.
OPEN_LOOP_HORIZON = 8

# Reply to a `sync` exchange: deliberately tiny, so both directions are ~one small packet.
_SYNC_REPLY = {"probe": "sync"}


def _load_constants() -> types.ModuleType:
    """Loads `examples/umi_rizon10/constants.py` -- one directory up, and not part of a package.

    Same module `main.py` imports. It needs `openpi.shared.se3`, which on the robot machine comes
    from `PYTHONPATH=<openpi>/src`.
    """
    path = pathlib.Path(__file__).resolve().parent.parent / "constants.py"
    spec = importlib.util.spec_from_file_location("umi_rizon10_constants", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation_payload(consts: types.ModuleType, prompt: str, rng: np.random.Generator) -> dict:
    """The dict `main.py:_observe()` sends, with shapes read out of `constants.features()`."""
    features = consts.features()
    return {
        "observation/wrist_image": rng.integers(256, size=tuple(features["wrist_image"]["shape"]), dtype=np.uint8),
        "observation/state": rng.random(tuple(features["state"]["shape"])).astype(np.float32),
        "prompt": prompt,
    }


def _action_payload(consts: types.ModuleType, rng: np.random.Generator) -> dict:
    """The response the real server returns: absolute base-frame poses for the whole chunk."""
    features = consts.features()
    action_dim = next(iter(features["actions"]["shape"]))
    return {
        "actions": rng.random((consts.ACTION_HORIZON, action_dim)).astype(np.float32),
        "state": rng.random(tuple(features["state"]["shape"])).astype(np.float32),
        "server_timing": {"infer_ms": 0.0},
    }


# --- server side (runs on the H100) ---------------------------------------------------------


@dataclasses.dataclass
class Serve:
    # Interface to bind. 0.0.0.0 accepts connections from the robot machine.
    host: str = "0.0.0.0"
    # Defaults to 8001 so this coexists with `scripts/serve_policy.py` on 8000.
    port: int = 8001
    # Sleep this long before replying, to emulate inference occupancy. The more faithful test is to
    # leave this at 0 and instead run the real policy server under load at the same time, which
    # captures actual NIC/CPU contention rather than simulating it.
    compute_ms: float = 0.0


async def _handle(ws: websockets.asyncio.server.ServerConnection, consts: types.ModuleType, compute_ms: float) -> None:
    packer = msgpack_numpy.Packer()
    rng = np.random.default_rng(0)
    compute_s = compute_ms / 1000.0
    reply = _action_payload(consts, rng)

    # Mirror `WebsocketPolicyServer._handler`: exactly one metadata frame on connect.
    await ws.send(packer.pack({"probe": "measure_command_latency", "compute_ms": compute_ms}))
    logger.info("Connection from %s opened", ws.remote_address)

    while True:
        try:
            raw = await ws.recv()
            t1 = time.time()
            request = msgpack_numpy.unpackb(raw)
            t1_decoded = time.time()

            marker = request.get("_probe") if isinstance(request, dict) else None
            if marker is None:
                # A real WebsocketClientPolicy is talking to us (`probe --verify_client_rtt`). It
                # sends one frame and expects exactly one back, so skip the trailer.
                if compute_s:
                    await asyncio.sleep(compute_s)
                await ws.send(packer.pack(reply))
                continue

            if compute_s:
                await asyncio.sleep(compute_s)

            # Pack *before* stamping t2, so t2 is the moment the bytes are handed to the kernel --
            # that is what "the command was sent from the H100" means. t2 cannot live inside the
            # frame it describes, so it follows in a tiny trailer frame; writing the trailer after
            # the payload cannot delay the payload, whose bytes are already queued.
            blob = packer.pack(_SYNC_REPLY if marker["kind"] == "sync" else reply)
            t2 = time.time()
            await ws.send(blob)
            t2_post = time.time()
            await ws.send(
                packer.pack(
                    {
                        "seq": marker["seq"],
                        "t1": t1,
                        "t1_decoded": t1_decoded,
                        "t2": t2,
                        "t2_post": t2_post,
                    }
                )
            )
        except websockets.ConnectionClosed:
            logger.info("Connection from %s closed", ws.remote_address)
            break


def _run_serve(args: Serve) -> None:
    consts = _load_constants()

    async def run() -> None:
        async def handler(ws: websockets.asyncio.server.ServerConnection) -> None:
            await _handle(ws, consts, args.compute_ms)

        # Same kwargs as `WebsocketPolicyServer.run`. `compression=None` is load-bearing:
        # permessage-deflate would change the wire sizes and defeat the realistic payloads.
        async with websockets.asyncio.server.serve(
            handler, args.host, args.port, compression=None, max_size=None
        ) as server:
            logger.info("Latency probe listening on %s:%d (compute_ms=%.1f)", args.host, args.port, args.compute_ms)
            await server.serve_forever()

    asyncio.run(run())


# --- probe side (runs on the robot machine) -------------------------------------------------


@dataclasses.dataclass
class Probe:
    # The H100 running `measure_command_latency.py serve`.
    host: str = "0.0.0.0"
    # Port that `serve` is listening on.
    port: int = 8001

    # Realistic-payload exchanges to measure.
    samples: int = 200
    # Tiny symmetric exchanges per clock-sync block. One block runs before and one after the
    # measurement, so the offset estimate can be drift-corrected.
    sync_samples: int = 200
    # Discarded realistic exchanges, to get past TCP slow start and first-touch allocation.
    warmup: int = 10

    # Request rate. Defaults to the real replan cadence, FPS / OPEN_LOOP_HORIZON = 2.5 Hz.
    pace_hz: float | None = None
    # Send as fast as the link allows instead of pacing. A saturation test, not a latency estimate.
    back_to_back: bool = False

    # Use this clock offset (server_clock - probe_clock, in seconds) instead of estimating one, e.g.
    # a value derived from `chronyc tracking` or PTP on both machines. The estimate is still
    # computed and reported alongside, for comparison.
    clock_offset_s: float | None = None

    # Prompt to send, so the payload size matches deployment.
    prompt: str = "pick up the tape and place it in the bin"

    # Also time this many real `WebsocketClientPolicy.infer()` calls, as a cross-check that this
    # raw-socket harness sees the same round trip the deployed client does. 0 disables.
    verify_client_rtt: int = 20

    # Write raw samples and statistics here as JSON.
    out: pathlib.Path | None = None


def _exchange(ws: websockets.sync.client.ClientConnection, packer, payload: dict, kind: str, seq: int) -> dict:
    """One request/response round trip, returning the raw timestamps.

    Cross-machine stamps are `time.time()` (CLOCK_REALTIME) because they have to be comparable
    across hosts. Purely local durations use `time.perf_counter()`.
    """
    request = {**payload, "_probe": {"kind": kind, "seq": seq}}

    pack_start = time.perf_counter()
    blob = packer.pack(request)
    pack_ms = (time.perf_counter() - pack_start) * 1e3

    # Stamp t0 after packing, mirroring t2 on the server and the real client, which also packs
    # before it starts sending (`WebsocketClientPolicy.infer`).
    t0 = time.time()
    ws.send(blob)

    raw = ws.recv()
    # `recv()` returns once the whole message is reassembled, so this is "last byte arrived", which
    # is the right reading of "when the robot computer receives the command".
    t3 = time.time()
    msgpack_numpy.unpackb(raw)
    t4 = time.time()
    trailer = msgpack_numpy.unpackb(ws.recv())

    return {
        "seq": seq,
        "t0": t0,
        "t1": trailer["t1"],
        "t1_decoded": trailer["t1_decoded"],
        "t2": trailer["t2"],
        "t2_post": trailer["t2_post"],
        "t3": t3,
        "t4": t4,
        "request_bytes": len(blob),
        "response_bytes": len(raw),
        "client_pack_ms": pack_ms,
    }


def _clock_sync(ws, packer, count: int, seq_start: int) -> dict:
    """Estimates the clock offset from `count` tiny symmetric exchanges.

    With `theta = server_clock - probe_clock`, NTP's estimator over one exchange is

        rtt   = (t3 - t0) - (t2 - t1)
        theta = ((t1 - t0) + (t2 - t3)) / 2

    Both are exact only when the two directions are equally fast. Sync exchanges are tiny in *both*
    directions to make that assumption as defensible as possible; the residual bias is
    (d_up - d_down) / 2, bounded by rtt / 2, which is the error bar the report quotes. The
    minimum-rtt sample is the least queue-contaminated, hence the standard choice.
    """
    samples = [_exchange(ws, packer, {}, "sync", seq_start + i) for i in range(count)]
    rtts = [(s["t3"] - s["t0"]) - (s["t2"] - s["t1"]) for s in samples]
    best = int(np.argmin(rtts))
    s = samples[best]
    return {
        "offset_s": ((s["t1"] - s["t0"]) + (s["t2"] - s["t3"])) / 2,
        "min_rtt_s": rtts[best],
        "median_rtt_s": float(np.median(rtts)),
        "at_s": (s["t0"] + s["t3"]) / 2,
        "n": count,
        "request_bytes": s["request_bytes"],
        "response_bytes": s["response_bytes"],
    }


def _metrics(s: dict, offset_s: float) -> dict:
    """Per-sample derived latencies, in milliseconds."""
    return {
        # What the deployed client experiences around one `infer()` call, server work included.
        "round_trip_total_ms": (s["t3"] - s["t0"]) * 1e3,
        # Round trip with the server's own handling removed -- pure network, both directions.
        "network_rtt_ms": ((s["t3"] - s["t0"]) - (s["t2"] - s["t1"])) * 1e3,
        "uplink_ms": (s["t1"] - s["t0"] - offset_s) * 1e3,
        "server_decode_ms": (s["t1_decoded"] - s["t1"]) * 1e3,
        "server_handling_ms": (s["t2"] - s["t1"]) * 1e3,
        "server_send_syscall_ms": (s["t2_post"] - s["t2"]) * 1e3,
        # The number this script exists for.
        "downlink_wire_ms": (s["t3"] - s["t2"] + offset_s) * 1e3,
        "client_decode_ms": (s["t4"] - s["t3"]) * 1e3,
        # ... and the same thing measured to the point where the control loop can use the array.
        "downlink_decoded_ms": (s["t4"] - s["t2"] + offset_s) * 1e3,
        "client_pack_ms": s["client_pack_ms"],
    }


_METRIC_ORDER = (
    "round_trip_total_ms",
    "network_rtt_ms",
    "uplink_ms",
    "server_decode_ms",
    "server_handling_ms",
    "server_send_syscall_ms",
    "downlink_wire_ms",
    "client_decode_ms",
    "downlink_decoded_ms",
    "client_pack_ms",
)


def _stats(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    q = np.quantile(arr, [0.5, 0.9, 0.95, 0.99])
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p50": float(q[0]),
        "p90": float(q[1]),
        "p95": float(q[2]),
        "p99": float(q[3]),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


# Block-drawing characters render a far more legible histogram, but a robot machine reached over
# ssh with LANG=C would raise UnicodeEncodeError on the very last line of a two-minute run.
_FULL, _PARTIAL = (
    ("\u2588", "\u258f\u258e\u258d\u258c\u258b\u258a\u2589")
    if ((sys.stdout.encoding or "").lower().replace("-", "").startswith("utf"))
    else ("#", "")
)


def _histogram(values: list[float], bins: int = 12, width: int = 36) -> list[str]:
    """A dependency-free horizontal histogram -- matplotlib is not in the robot-side env."""
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    peak = max(int(counts.max()), 1)
    # Enough decimals that adjacent bin edges stay distinguishable: a sub-millisecond LAN and a
    # 100 ms WAN both get readable labels.
    step = float(edges[1] - edges[0])
    decimals = int(np.clip(1 - np.floor(np.log10(step)) if step > 0 else 1, 0, 4))
    lines = []
    for count, lo, hi in zip(counts, edges[:-1], edges[1:], strict=True):
        eighths = round(int(count) / peak * width * 8)
        full, frac = divmod(eighths, 8)
        bar = _FULL * full + (_PARTIAL[frac - 1] if frac and _PARTIAL else "")
        lines.append(f"  {lo:8.{decimals}f} - {hi:<8.{decimals}f} {count:4d}  {bar}".rstrip())
    return lines


def _collect_load(ws, packer, payload: dict, count: int, pace_hz: float | None, into: list) -> None:
    """Runs `count` realistic exchanges, appending into `into` so a Ctrl-C keeps partial results."""
    start = time.perf_counter()
    for i in range(count):
        if pace_hz:
            delay = start + i / pace_hz - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        into.append(_exchange(ws, packer, payload, "load", i))


def _verify_client_rtt(host: str, port: int, count: int, payload: dict, pace_hz: float | None) -> list:
    """Times real `WebsocketClientPolicy.infer()` calls against the probe server.

    Paced identically to the measurement loop. That matters more than it looks: at the deployed
    2.5 Hz the connection sits idle for 400 ms between requests, and Linux's
    `tcp_slow_start_after_idle` shrinks the congestion window over that gap, so comparing against a
    back-to-back loop would be comparing a cold connection to a warm one.
    """
    client = websocket_client_policy.WebsocketClientPolicy(host, port)
    for _ in range(3):
        client.infer(payload)
    rtts = []
    start = time.perf_counter()
    for i in range(count):
        if pace_hz:
            delay = start + i / pace_hz - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        call_start = time.perf_counter()
        client.infer(payload)
        rtts.append((time.perf_counter() - call_start) * 1e3)
    return rtts


def _run_probe(args: Probe) -> None:
    consts = _load_constants()
    pace_hz = (
        None if args.back_to_back else (args.pace_hz if args.pace_hz is not None else consts.FPS / OPEN_LOOP_HORIZON)
    )
    rng = np.random.default_rng(0)
    payload = _observation_payload(consts, args.prompt, rng)
    packer = msgpack_numpy.Packer()

    uri = f"ws://{args.host}:{args.port}"
    logger.info("Connecting to %s", uri)
    interrupted = False
    load: list = []

    # Same kwargs `WebsocketClientPolicy._wait_for_server` uses. No socket tuning on purpose:
    # whatever `websockets.sync.client` does by default is what the deployed client does.
    with websockets.sync.client.connect(uri, compression=None, max_size=None) as ws:
        logger.info("Server metadata: %s", msgpack_numpy.unpackb(ws.recv()))

        for i in range(args.warmup):
            _exchange(ws, packer, payload, "load", -1 - i)

        logger.info("Clock sync (%d tiny exchanges)...", args.sync_samples)
        sync_start = _clock_sync(ws, packer, args.sync_samples, 0)

        logger.info(
            "Measuring %d realistic exchanges at %s...",
            args.samples,
            "full rate" if pace_hz is None else f"{pace_hz:.2f} Hz",
        )
        try:
            _collect_load(ws, packer, payload, args.samples, pace_hz, load)
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Interrupted after %d samples -- reporting what was collected.", len(load))

        sync_end = _clock_sync(ws, packer, args.sync_samples, args.sync_samples)

    if not load:
        raise RuntimeError("No samples collected.")

    span_s = sync_end["at_s"] - sync_start["at_s"]
    drift_ppm = (sync_end["offset_s"] - sync_start["offset_s"]) / span_s * 1e6 if span_s > 0 else 0.0
    # The error bar on every one-way number: the sync offset can be biased by up to half the
    # smallest observed tiny-packet round trip, if that path is maximally asymmetric.
    error_bar_ms = min(sync_start["min_rtt_s"], sync_end["min_rtt_s"]) / 2 * 1e3

    def offset_at(t: float) -> float:
        if args.clock_offset_s is not None:
            return args.clock_offset_s
        if span_s <= 0:
            return sync_start["offset_s"]
        frac = (t - sync_start["at_s"]) / span_s
        return sync_start["offset_s"] + (sync_end["offset_s"] - sync_start["offset_s"]) * frac

    metrics = [_metrics(s, offset_at(s["t0"])) for s in load]
    stats = {name: _stats([m[name] for m in metrics]) for name in _METRIC_ORDER}

    client_rtts = (
        _verify_client_rtt(args.host, args.port, args.verify_client_rtt, payload, pace_hz)
        if args.verify_client_rtt > 0
        else []
    )

    # --- report -----------------------------------------------------------------------------
    # One number is the point of this script; everything else is either a validity check or
    # detail that belongs in --out.
    wire, usable = stats["downlink_wire_ms"], stats["downlink_decoded_ms"]
    up_b, down_b = load[0]["request_bytes"], load[0]["response_bytes"]
    # The asymmetry is why the round trip cannot simply be halved.
    ratio = f"up:down = {up_b / down_b:.0f}:1" if up_b >= down_b else f"up:down = 1:{down_b / up_b:.0f}"
    print("\n=== downlink: inference server -> robot machine ===")
    print(
        f"{uri}   {len(load)} samples{' (INTERRUPTED)' if interrupted else ''} @ "
        f"{'back-to-back' if pace_hz is None else f'{pace_hz:.2f} Hz'}   "
        f"request {up_b} B / response {down_b} B ({ratio})"
    )
    print(f"\n  H100 send -> bytes arrived   {wire['p50']:7.1f} ms   p95 {wire['p95']:6.1f}   max {wire['max']:6.1f}")
    print(
        f"  H100 send -> chunk usable    {usable['p50']:7.1f} ms   p95 {usable['p95']:6.1f}   max {usable['max']:6.1f}"
    )
    print(f"  +/- {error_bar_ms:.1f} ms clock-sync error bar on both (half the smallest tiny-packet rtt)")
    print(f"\n  downlink_lag_s = {usable['p50'] / 1e3:.4f}   <- for examples/umi_rizon10/calibration/*.yaml")

    print(f"\n  distribution of 'chunk usable' (ms), n={len(metrics)}")
    for line in _histogram([m["downlink_decoded_ms"] for m in metrics]):
        print(line)

    print(
        f"\nfor reference: uplink {stats['uplink_ms']['p50']:.1f} ms, msgpack decode "
        f"{stats['client_decode_ms']['p50']:.1f} ms, server handling "
        f"{stats['server_handling_ms']['p50']:.1f} ms, clock drift {drift_ppm:+.0f} ppm over {span_s:.0f} s"
    )

    problems = []
    negative = sum(1 for m in metrics if m["downlink_wire_ms"] < 0)
    # Only meaningful for a positive measurement -- a negative one is already covered below, and
    # comparing against half of a negative number would flag every run.
    weak_split = usable["p50"] > 0 and error_bar_ms > 0.5 * usable["p50"]
    if negative:
        problems.append(
            f"{negative}/{len(metrics)} samples have a NEGATIVE downlink, so the clock offset is "
            "biased past its error bar (asymmetric routing, or the clock stepped mid-run). Treat "
            "the one-way split as unreliable."
        )
    if weak_split:
        problems.append(
            f"the error bar (+/- {error_bar_ms:.1f} ms) exceeds half the measured value, so the "
            "one-way split is only weakly determined."
        )
    if abs(drift_ppm) > 100:
        problems.append(f"{drift_ppm:+.0f} ppm of clock drift is large; check ntp/chrony on both hosts.")
    if client_rtts:
        # Both are paced identically, so a large gap means this harness is not measuring the path
        # the deployed client actually takes.
        gap = abs(_stats(client_rtts)["p50"] - stats["round_trip_total_ms"]["p50"])
        if gap > 0.25 * stats["round_trip_total_ms"]["p50"]:
            problems.append(
                f"a real WebsocketClientPolicy.infer() round trip differs from this harness by "
                f"{gap:.1f} ms at p50; the harness may not be on the deployed path."
            )
    for problem in problems:
        print(f"\nWARNING: {problem}")
    if negative or weak_split:
        print("  -> get an external offset (`chronyc tracking` on both hosts against a common NTP")
        print("     server, or `ptp4l`) and pass it as --clock_offset_s.")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "setup": {
                        "uri": uri,
                        "samples": len(load),
                        "interrupted": interrupted,
                        "pace_hz": pace_hz,
                        "warmup": args.warmup,
                        "prompt": args.prompt,
                        "request_bytes": up_b,
                        "response_bytes": down_b,
                        "fps": consts.FPS,
                        "action_horizon": consts.ACTION_HORIZON,
                    },
                    "clock": {
                        "sync_start": sync_start,
                        "sync_end": sync_end,
                        "drift_ppm": drift_ppm,
                        "error_bar_ms": error_bar_ms,
                        "override_s": args.clock_offset_s,
                    },
                    "stats": stats,
                    "client_rtt_ms": client_rtts,
                    "downlink_lag_s": usable["p50"] / 1e3,
                    "samples_raw": load,
                    "metrics_raw": metrics,
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.out}")


def main(cmd: Serve | Probe) -> None:
    if isinstance(cmd, Serve):
        _run_serve(cmd)
    else:
        _run_probe(cmd)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    main(tyro.extras.subcommand_cli_from_dict({"serve": Serve, "probe": Probe}))
