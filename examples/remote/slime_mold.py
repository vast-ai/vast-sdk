from vastai.serverless.remote.endpoint import *
import base64



@benchmark(
    endpoint_name="slime_mold_pytorch",
    dataset=[{
        "width": 512,
        "height": 512,
        "steps": 400000,
        "n_agents": 50000
    }]
)
@remote(endpoint_name="slime_mold_pytorch")
async def render_slime_mold(width: int = 512,
                            height: int = 512,
                            steps: int = 4000,
                            n_agents: int = 50000,
                            sensor_distance: float = 9.0,
                            sensor_angle: float = 0.5,
                            rotation_angle: float = 0.3,
                            step_size: float = 1.0,
                            decay: float = 0.96,
                            random_rotation: float = 0.2):
    """
    Physarum-style slime mold simulation using PyTorch.

    Rules (based on Jeff Jones 2010 model & common GPU ports):
      1. Each agent senses trail ahead, left, and right.
      2. Turn toward the direction with strongest trail.
      3. If sensors are ~equal, apply random turn.
      4. Move forward, deposit trail.
      5. Diffuse + decay trail (blur + multiply).
    Tries GPU, falls back to CPU if CUDA is incompatible.
    """
    import math
    import torch as t
    import torch.nn.functional as F
    from io import BytesIO
    from PIL import Image

    def _simulate(device: str):
        # Trail field
        trail = t.zeros((height, width), device=device, dtype=t.float32)

        # Agent positions & headings
        agents_pos = t.rand((n_agents, 2), device=device, dtype=t.float32)
        agents_pos[:, 0] *= width
        agents_pos[:, 1] *= height
        agents_angle = t.rand((n_agents,), device=device) * 2.0 * math.pi

        sensor_dist = t.tensor(sensor_distance, device=device)
        sensor_ang = t.tensor(sensor_angle, device=device)
        rot_ang = t.tensor(rotation_angle, device=device)
        step = t.tensor(step_size, device=device)
        rand_rot_strength = t.tensor(random_rotation, device=device)

        def sample_sensor(offset_angle: t.Tensor):
            """
            Sample trail at positions ahead of each agent with an angle offset.
            offset_angle: scalar or tensor broadcastable to (n_agents,)
            """
            ang = agents_angle + offset_angle
            dx = t.cos(ang) * sensor_dist
            dy = t.sin(ang) * sensor_dist
            sx = (agents_pos[:, 0] + dx).long().clamp(0, width - 1)
            sy = (agents_pos[:, 1] + dy).long().clamp(0, height - 1)
            return trail[sy, sx]

        with t.no_grad():
            for _ in range(steps):
                # --- 1. Sense ---
                center = sample_sensor(0.0)
                left = sample_sensor(sensor_ang)
                right = sample_sensor(-sensor_ang)

                # --- 2. Decide rotation ---
                angle_change = t.zeros_like(agents_angle)

                # Turn toward strongest sensor
                turn_left = (left > center) & (left > right)
                turn_right = (right > center) & (right > left)

                angle_change = t.where(turn_left,
                                       angle_change + rot_ang,
                                       angle_change)
                angle_change = t.where(turn_right,
                                       angle_change - rot_ang,
                                       angle_change)

                # Where all sensors are ~equal -> wander randomly
                eps = 1e-5
                all_close = (
                    (center - left).abs() < eps
                ) & (
                    (center - right).abs() < eps
                ) & (
                    (left - right).abs() < eps
                )

                # Random small rotation (uniform in [-rand_rot_strength, rand_rot_strength])
                rand_rot = (t.rand_like(agents_angle) * 2.0 - 1.0) * rand_rot_strength
                angle_change = t.where(all_close, rand_rot, angle_change + 0.25 * rand_rot)

                # Apply rotation
                agents_angle = agents_angle + angle_change

                # --- 3. Move ---
                dx = t.cos(agents_angle) * step
                dy = t.sin(agents_angle) * step
                agents_pos[:, 0] += dx
                agents_pos[:, 1] += dy

                # Wrap around edges (toroidal space)
                agents_pos[:, 0] = agents_pos[:, 0].remainder(width)
                agents_pos[:, 1] = agents_pos[:, 1].remainder(height)

                # --- 4. Deposit trail ---
                ix = agents_pos[:, 0].long().clamp(0, width - 1)
                iy = agents_pos[:, 1].long().clamp(0, height - 1)
                trail.index_put_((iy, ix),
                                 t.ones_like(ix, dtype=trail.dtype),
                                 accumulate=True)

                # --- 5. Diffuse & decay trail ---
                trail = F.avg_pool2d(
                    trail[None, None, :, :],
                    kernel_size=3,
                    stride=1,
                    padding=1
                )[0, 0]
                trail *= decay

        # Normalize to 0..1
        max_val = trail.max()
        if max_val > 0:
            img_f = trail / max_val
        else:
            img_f = trail

        # Invert for nicer viewing (bright trails on dark)
        img_f = 1.0 - img_f
        img_f = img_f.clamp(0.0, 1.0)

        # To 3-channel grayscale image
        img_np = (img_f * 255).byte().cpu().numpy()
        img_rgb = t.stack(
            [t.from_numpy(img_np)] * 3,
            dim=-1
        ).numpy()

        pil_img = Image.fromarray(img_rgb, mode="RGB")
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        return base64.b64encode(png_bytes).decode("ascii")

    import torch as t
    device = "cuda:0" if t.cuda.is_available() else "cpu"

    try:
        png_b64 = _simulate(device)
        device_used = device
    except RuntimeError as e:
        msg = str(e)
        if (
            "no kernel image is available for execution on the device" in msg
            or "CUDA error" in msg
        ):
            png_b64 = _simulate("cpu")
            device_used = "cpu"
        else:
            raise

    return {
        "width": width,
        "height": height,
        "steps": steps,
        "n_agents": n_agents,
        "png_base64": png_b64,
        "device_used": device_used,
    }

# Define the endpoint using the official PyTorch image
slime_ep = Endpoint(
    name="slime_mold_pytorch",
    image_name="pytorch/pytorch"
)
slime_ep.uv_pip_install("Pillow")

slime_ep.ready()
