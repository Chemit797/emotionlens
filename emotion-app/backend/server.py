"""EmotionLens · FastAPI WebSocket server.

Routes webcam frames through EngineFER, then dispatches to the active Lens.
Supports the shared control protocol: {type:"control", mode, action, duration}.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import asyncio
import json
import os
import base64
import numpy as np
import cv2

from backend.engine_fer import EngineFER
from backend.lenses.l1_cafeteria import CafeteriaMoodLens
from backend.lenses.l2_code_red import CodeRedLens
from backend.lenses.l3_audience import AudienceReactionsLens
from backend.lenses.l4_speech import SpeechCoachLens
from backend.lenses.l5_mimic import MimicGameLens

# Resolve frontend directory relative to this file (not cwd)
_FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

app = FastAPI()
app.mount("/frontend", StaticFiles(directory=_FRONTEND_DIR), name="frontend")


@app.get("/")
def read_root():
    return RedirectResponse(url="/frontend/index.html")


# ── Global engine (shared across all connections) ──────────────
engine = EngineFER()

# ── Lens registry ──────────────────────────────────────────────
LENS_CLASSES = {
    "m1": CafeteriaMoodLens,
    "m2": CodeRedLens,
    "m3": AudienceReactionsLens,
    "m4": SpeechCoachLens,
    "m5": MimicGameLens,
}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Each connection gets its own lens instances (per-mode state isolation)
    lenses: dict = {}
    for mode_id, cls in LENS_CLASSES.items():
        lenses[mode_id] = cls()

    current_mode = "m0"

    # Frame-drop queue
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    async def frame_processor():
        """Background task: consume latest frame, run engine + active lens, send results."""
        while True:
            ts, img, mode = await frame_queue.get()
            try:
                h, w = img.shape[:2]
                faces, rendered_frame = engine.process_frame(img)

                # Feed all lenses so their buffers stay in sync
                mode_out = {}
                if mode in lenses:
                    lens = lenses[mode]
                    lens.add_frame(ts, faces)
                    mode_out = lens.get_output(ts)

                # M4: auto-trigger LLM finalization when timer expires
                if mode == "m4" and lens.state == "generating" and lens._result is None:
                    asyncio.create_task(_finalize_m4(lens))

                # Encode rendered frame
                _, buffer = cv2.imencode('.jpg', rendered_frame)
                rendered_b64 = base64.b64encode(buffer).decode('utf-8')

                await websocket.send_json({
                    "type": "result",
                    "ts": ts,
                    "faces": faces,
                    "mode": mode,
                    "mode_output": mode_out,
                    "frame_width": w,
                    "frame_height": h,
                    "rendered_b64": rendered_b64,
                    "model": getattr(engine, "model_name", "unknown"),
                })
            except WebSocketDisconnect:
                return
            except Exception as e:
                print(f"Processor error: {e}")

    processor_task = asyncio.create_task(frame_processor())

    try:
        while True:
            text_data = await websocket.receive_text()
            data = json.loads(text_data)

            # ── Control messages (start/stop/reset) ──
            if data.get("type") == "control":
                mode = data.get("mode", current_mode)
                action = data.get("action", "")
                duration = data.get("duration", None)
                if mode in lenses:
                    lenses[mode].handle_control(action, duration)
                    # M4: after timer expires, needs async LLM call
                    if mode == "m4" and action == "start":
                        pass  # Will be handled when timer expires in get_output
                continue

            # ── Mode switch ──
            if data.get("type") == "set_mode":
                current_mode = data["mode"]
                continue

            # ── Frame ──
            if data.get("type") == "frame":
                ts = data["ts"]
                b64 = data["data"].split(",")[1]
                img_data = base64.b64decode(b64)
                np_arr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img is not None:
                    # Drop stale frame, enqueue latest
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    frame_queue.put_nowait((ts, img, current_mode))
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WS error: {e}")
    finally:
        processor_task.cancel()


async def _finalize_m4(lens: SpeechCoachLens):
    """Run M4 finalization (metrics + LLM) in background."""
    try:
        await lens.finalize()
    except Exception as e:
        print(f"M4 finalize error: {e}")
        lens.state = "done"
