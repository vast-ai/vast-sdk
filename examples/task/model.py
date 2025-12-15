#!/usr/bin/env python3
"""
Single-worker PyTorch trainer + aiohttp webserver.

Endpoints (all POST, JSON):
  1) /start_task  -> starts a new MNIST training run (CPU), returns task_id
  2) /status      -> returns current task state + metrics
  3) /cancel_task -> cancels the current task (best-effort), returns state

Notes:
- This backend manages exactly one active task at a time.
- Training runs in a background thread, while the webserver remains responsive.
- Status is kept in a shared state object guarded by a lock.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from aiohttp import web

# ---- PyTorch / TorchVision ----
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# Training model + helpers
# -----------------------------

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 14x14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 7x7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def now_s() -> float:
    return time.time()


# -----------------------------
# Task lifecycle + status
# -----------------------------

@dataclass
class TaskConfig:
    epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    max_train_batches_per_epoch: int = 200   # keep it small / fast
    max_val_batches: int = 50
    seed: int = 1337
    data_dir: str = "./data"
    num_workers: int = 0  # keep deterministic + simple


@dataclass
class TaskStatus:
    task_id: Optional[str] = None
    state: str = "idle"  # idle|running|completed|failed|canceled
    message: str = ""
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    # Progress / metrics
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    train_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None

    # Last update + config snapshot
    last_update_at: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)

    # Error details (if failed)
    error_type: Optional[str] = None
    error: Optional[str] = None


class TaskManager:
    """
    Single-worker task manager:
      - at most one active training task at a time
      - training runs in a background thread
      - cancellation via threading.Event
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._status = TaskStatus()
        self._thread: Optional[threading.Thread] = None
        self._cancel_event: Optional[threading.Event] = None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._status, default=lambda o: o.__dict__))

    def can_start(self) -> Tuple[bool, str]:
        with self._lock:
            if self._status.state == "running":
                return False, "A task is already running"
            return True, ""

    def start(self, cfg: TaskConfig) -> str:
        ok, msg = self.can_start()
        if not ok:
            raise RuntimeError(msg)

        task_id = str(uuid.uuid4())
        cancel_event = threading.Event()

        with self._lock:
            self._status = TaskStatus(
                task_id=task_id,
                state="running",
                message="Task started",
                created_at=now_s(),
                started_at=now_s(),
                epoch=0,
                step=0,
                total_steps=0,
                last_update_at=now_s(),
                config=cfg.__dict__.copy(),
            )
            self._cancel_event = cancel_event

        t = threading.Thread(
            target=self._train_entrypoint,
            name=f"trainer-{task_id}",
            args=(task_id, cfg, cancel_event),
            daemon=True,
        )
        self._thread = t
        t.start()
        return task_id

    def cancel(self) -> Dict[str, Any]:
        with self._lock:
            if self._status.state != "running":
                return self.snapshot()
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._status.message = "Cancellation requested"
            self._status.last_update_at = now_s()
        return self.snapshot()

    def _set_status_update(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._status, k, v)
            self._status.last_update_at = now_s()

    def _train_entrypoint(self, task_id: str, cfg: TaskConfig, cancel_event: threading.Event) -> None:
        try:
            run_training(task_id=task_id, cfg=cfg, cancel_event=cancel_event, report=self._set_status_update)

            if cancel_event.is_set():
                self._set_status_update(
                    state="canceled",
                    message="Task canceled",
                    finished_at=now_s(),
                )
            else:
                self._set_status_update(
                    state="completed",
                    message="Task completed",
                    finished_at=now_s(),
                )
        except Exception as e:
            self._set_status_update(
                state="failed",
                message="Task failed",
                finished_at=now_s(),
                error_type=type(e).__name__,
                error=str(e),
            )


# -----------------------------
# Training loop
# -----------------------------

def run_training(
    task_id: str,
    cfg: TaskConfig,
    cancel_event: threading.Event,
    report,
) -> None:
    # Determinism-ish
    torch.manual_seed(cfg.seed)

    device = torch.device("cpu")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=tfm)
    val_ds = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False
    )

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    steps_per_epoch = min(cfg.max_train_batches_per_epoch, len(train_loader))
    total_steps = steps_per_epoch * cfg.epochs

    report(
        total_steps=total_steps,
        message="Training initialized",
        epoch=0,
        step=0,
        train_loss=None,
        train_acc=None,
        val_loss=None,
        val_acc=None,
    )

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        if cancel_event.is_set():
            report(message=f"Canceled before epoch {epoch}", epoch=epoch, step=global_step)
            return

        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            if batch_idx > cfg.max_train_batches_per_epoch:
                break
            if cancel_event.is_set():
                report(message=f"Canceled during epoch {epoch}", epoch=epoch, step=global_step)
                return

            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            b_loss = loss.item()
            b_acc = accuracy(logits.detach(), y)

            running_loss += b_loss
            running_acc += b_acc
            n_batches += 1
            global_step += 1

            if batch_idx % 10 == 0 or batch_idx == steps_per_epoch:
                report(
                    epoch=epoch,
                    step=global_step,
                    message=f"Training epoch {epoch}/{cfg.epochs} batch {batch_idx}/{steps_per_epoch}",
                    train_loss=running_loss / max(1, n_batches),
                    train_acc=running_acc / max(1, n_batches),
                )

        # Validation (small)
        model.eval()
        v_loss_sum = 0.0
        v_acc_sum = 0.0
        v_batches = 0
        with torch.no_grad():
            for v_idx, (x, y) in enumerate(val_loader, start=1):
                if v_idx > cfg.max_val_batches:
                    break
                if cancel_event.is_set():
                    report(message=f"Canceled during validation epoch {epoch}", epoch=epoch, step=global_step)
                    return

                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)

                v_loss_sum += loss.item()
                v_acc_sum += accuracy(logits, y)
                v_batches += 1

        report(
            epoch=epoch,
            step=global_step,
            message=f"Validation epoch {epoch}/{cfg.epochs} complete",
            val_loss=v_loss_sum / max(1, v_batches),
            val_acc=v_acc_sum / max(1, v_batches),
        )


# -----------------------------
# aiohttp server
# -----------------------------

async def json_request(request: web.Request) -> Dict[str, Any]:
    if request.content_type and "application/json" in request.content_type:
        try:
            return await request.json()
        except Exception:
            return {}
    return {}


def make_app(manager: TaskManager) -> web.Application:
    app = web.Application()

    async def start_task(request: web.Request) -> web.Response:
        payload = await json_request(request)

        # Allow overrides via JSON; keep defaults safe.
        cfg = TaskConfig(
            epochs=int(payload.get("epochs", 2)),
            batch_size=int(payload.get("batch_size", 64)),
            lr=float(payload.get("lr", 1e-3)),
            max_train_batches_per_epoch=int(payload.get("max_train_batches_per_epoch", 200)),
            max_val_batches=int(payload.get("max_val_batches", 50)),
            seed=int(payload.get("seed", 1337)),
            data_dir=str(payload.get("data_dir", "./data")),
            num_workers=int(payload.get("num_workers", 0)),
        )

        try:
            task_id = manager.start(cfg)
            return web.json_response({"ok": True, "task_id": task_id, "status": manager.snapshot()})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e), "status": manager.snapshot()}, status=409)

    async def status(request: web.Request) -> web.Response:
        # No body required; POST for uniformity.
        _ = await json_request(request)
        return web.json_response({"ok": True, "status": manager.snapshot()})

    async def cancel_task(request: web.Request) -> web.Response:
        _ = await json_request(request)
        st = manager.cancel()
        return web.json_response({"ok": True, "status": st})

    app.router.add_post("/start_task", start_task)
    app.router.add_post("/status", status)
    app.router.add_post("/cancel_task", cancel_task)

    # Basic liveness
    async def health(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    app.router.add_get("/health", health)
    return app


async def _run_server(host: str, port: int) -> None:
    manager = TaskManager()
    app = make_app(manager)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    print("Model Server Running")
    stop_event = asyncio.Event()

    def _handle_sig(*_args: Any) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            # Windows fallback
            signal.signal(sig, lambda *_: stop_event.set())

    await stop_event.wait()

    # Best-effort cancel on shutdown
    manager.cancel()
    await runner.cleanup()


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    asyncio.run(_run_server(host, port))


if __name__ == "__main__":
    main()
