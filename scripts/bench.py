#!/usr/bin/env python3
"""Drive the ComfyUI API and record stage/step timings for MiniMax-H3 Ref2VA.

Every run uses a fresh client_id, a different seed and a prompt whose [Shot N]
tag changes so ComfyUI's node-output cache cannot satisfy the conditioning node.
"""
import argparse
import asyncio
import glob
import json
import os
import statistics
import sys
import time
import uuid

import aiohttp

STAGE_NODES = {
    "unet_load": ["1"],
    "lora_apply": ["2"],
    "clip_load": ["3"],
    "vae_load": ["4", "5"],
    "conditioning_te_ref_vae": ["6"],
    "sampler": ["11"],
    "vae_decode_video": ["12"],
    "vae_decode_audio": ["13"],
    "video_encode": ["14"],
    "save_video": ["15"],
    "image_load": ["20"],
}


def load_workflow(path, prompt, seed):
    with open(path, encoding="utf-8") as f:
        wf = json.load(f)
    wf["6"]["inputs"]["prompt"] = prompt
    wf["10"]["inputs"]["noise_seed"] = seed
    return wf


def analyze(events, labels):
    """Convert a raw event list into stage durations and per-step times."""
    t_start = next((e["t"] for e in events if e["type"] == "execution_start"),
                   events[0]["t"] if events else 0.0)
    t_end = next((e["t"] for e in events if e["type"] == "execution_success"), None)
    node_start = []
    progress = []
    cached = []
    for e in events:
        d = e["data"]
        if e["type"] == "executing" and d.get("node") is not None:
            node_start.append((str(d["node"]), e["t"]))
        elif e["type"] == "progress":
            progress.append((str(d.get("node")), int(d.get("value", 0)), e["t"]))
        elif e["type"] == "execution_cached":
            cached.extend(str(n) for n in d.get("nodes", []))

    # Node durations: an executing event marks the start; the next node (or success) ends it.
    durations = {}
    for i, (node, t) in enumerate(node_start):
        end = node_start[i + 1][1] if i + 1 < len(node_start) else (t_end or t)
        durations.setdefault(node, []).append(end - t)

    stages = {}
    for name, nodes in STAGE_NODES.items():
        vals = [v for n in nodes for v in durations.get(n, [])]
        if vals:
            stages[name] = sum(vals)

    steps = []
    sampler_t = next((t for n, t in node_start if n == "11"), None)
    sampler_progress = sorted((v, t) for n, v, t in progress if n == "11")
    prev_t = sampler_t
    for value, t in sampler_progress:
        if prev_t is not None and t >= prev_t:
            steps.append({"step": value, "s": t - prev_t})
        prev_t = t

    return {
        "execution_start_s": t_start,
        "total_compute_s": (t_end - t_start) if t_end is not None else None,
        "stages_s": stages,
        "steps": steps,
        "step_median_s": statistics.median([s["s"] for s in steps]) if steps else None,
        "sampling_total_s": sum(s["s"] for s in steps) if steps else None,
        "cached_nodes": sorted(set(cached)),
        "node_durations_s": {k: sum(v) for k, v in sorted(durations.items())},
    }


async def run_once(session, base, workflow, label, seed, prompt, outdir, events_dir, timeout):
    client_id = str(uuid.uuid4())
    ws_url = base.replace("http", "ws", 1) + f"/ws?clientId={client_id}"
    ws = await session.ws_connect(ws_url, heartbeat=60, max_msg_size=0)
    started = time.monotonic()
    events = []
    error = None
    pid = None
    try:
        async with session.post(f"{base}/prompt",
                                json={"prompt": workflow, "client_id": client_id},
                                timeout=aiohttp.ClientTimeout(total=120)) as r:
            body = await r.text()
            if r.status != 200:
                raise RuntimeError(f"/prompt HTTP {r.status}: {body}")
            reply = json.loads(body)
        pid = reply.get("prompt_id")
        if not pid:
            raise RuntimeError(f"/prompt rejected: {reply}")
        print(f"[{label}] queued prompt_id={pid} seed={seed}", flush=True)

        while True:
            msg = await ws.receive(timeout=timeout)
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                etype = data.get("type")
                d = data.get("data", {})
                if d.get("prompt_id") not in (None, pid):
                    continue
                events.append({"t": time.monotonic() - started, "type": etype, "data": d})
                if etype == "execution_success" and d.get("prompt_id") == pid:
                    break
                if etype in ("execution_error", "execution_interrupted") and d.get("prompt_id") == pid:
                    error = d
                    break
            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                error = f"websocket {msg.type}"
                break
    finally:
        await ws.close()

    wall = time.monotonic() - started
    files = sorted(glob.glob(os.path.join(outdir, "**", "*.mp4"), recursive=True),
                   key=os.path.getmtime)
    newest = files[-1] if files and os.path.getmtime(files[-1]) >= started - 5 else None

    raw = {"label": label, "seed": seed, "prompt_id": pid,
           "wall_s": wall, "error": error, "events": events}
    os.makedirs(events_dir, exist_ok=True)
    with open(os.path.join(events_dir, f"{label}_{int(time.time())}.json"), "w") as f:
        json.dump(raw, f)
    if error:
        print(f"[{label}] ERROR: {json.dumps(error)[:800]}", flush=True)
        return {"label": label, "seed": seed, "wall_s": wall, "error": error}

    res = analyze(events, label)
    res.update({"label": label, "seed": seed, "wall_s": wall, "output": newest, "error": None})
    total = res["total_compute_s"]
    median = res["step_median_s"]
    print(f"[{label}] total={total if total is None else round(total, 1)}s "
          f"steps={[round(s['s'], 2) for s in res['steps']]} "
          f"median={median if median is None else round(median, 1)}s", flush=True)
    return res


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8190")
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--runs", default="cold:1001,warm:1002,warm:1003")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    base_prompt = open(args.prompt_file, encoding="utf-8").read().strip()
    runs = []
    for i, token in enumerate(args.runs.split(","), start=1):
        label, seed = token.split(":")
        runs.append((label, int(seed), i))

    results = []
    async with aiohttp.ClientSession() as session:
        for label, seed, shot in runs:
            # Vary the shot tag so the conditioning node is never served from cache.
            prompt = base_prompt.replace("[Shot 1]", f"[Shot {shot}]", 1)
            wf = load_workflow(args.workflow, prompt, seed)
            try:
                res = await run_once(session, args.base, wf, label, seed, prompt,
                                     args.outdir, args.events_dir, args.timeout)
            except Exception as exc:  # keep earlier runs even if one fails
                res = {"label": label, "seed": seed, "error": repr(exc)}
                print(f"[{label}] EXCEPTION: {exc!r}", flush=True)
            res["shot_tag"] = shot
            results.append(res)
            with open(args.results, "w") as f:
                json.dump({"runs": results}, f, indent=2)
            if res.get("error"):
                break

    return 0 if all(not r.get("error") for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
