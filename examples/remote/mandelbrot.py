from vastai.serverless.remote.endpoint import *
import base64


@benchmark(
    endpoint_name="mandelbrot_pytorch",
    dataset=[{"width": 1024, "height": 768, "max_iter": 200}]
)
@remote(endpoint_name="mandelbrot_pytorch")
async def render_mandelbrot(width: int = 1024, height: int = 768, max_iter: int = 200):
    import torch as t
    from io import BytesIO
    from PIL import Image

    def _render(device: str):
        re = t.linspace(-2.5, 1.0, width, device=device)
        im = t.linspace(-1.5, 1.5, height, device=device)
        c_re, c_im = t.meshgrid(re, im, indexing="xy")

        z_re = t.zeros_like(c_re)
        z_im = t.zeros_like(c_im)

        diverged = t.zeros_like(c_re, dtype=t.bool)
        iters = t.zeros_like(c_re, dtype=t.int32)

        with t.no_grad():
            for i in range(max_iter):
                # z = z^2 + c
                z_re2 = z_re * z_re - z_im * z_im + c_re
                z_im2 = 2.0 * z_re * z_im + c_im
                z_re, z_im = z_re2, z_im2

                mag2 = z_re * z_re + z_im * z_im
                newly_diverged = (~diverged) & (mag2 > 4.0)
                iters[newly_diverged] = i
                diverged |= newly_diverged

                if diverged.all():
                    break

            norm = iters.float() / max_iter  # 0..1
            norm = norm.clamp(0.0, 1.0)

            r = 9.0 * (1 - norm) * (norm ** 3)
            g = 15.0 * ((1 - norm) ** 2) * (norm ** 2)
            b = 8.5 * ((1 - norm) ** 3) * norm

            rgb = t.stack([r, g, b], dim=-1)

            inside_mask = ~diverged
            rgb[inside_mask] = 0.0

            rgb = rgb.clamp(0.0, 1.0)

            img = (rgb * 255).byte().permute(1, 0, 2).cpu().numpy()

        pil_img = Image.fromarray(img, mode="RGB")
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        return base64.b64encode(png_bytes).decode("ascii")

    import torch as t
    device = "cuda:0" if t.cuda.is_available() else "cpu"

    try:
        png_b64 = _render(device)
        device_used = device
    except RuntimeError as e:
        msg = str(e)
        if "no kernel image is available for execution on the device" in msg or "CUDA error" in msg:
            png_b64 = _render("cpu")
            device_used = "cpu"
        else:
            raise

    return {
        "width": width,
        "height": height,
        "max_iter": max_iter,
        "png_base64": png_b64,
        "device_used": device_used,
    }

fractal_ep = Endpoint(
    name="mandelbrot_pytorch",
    image_name="pytorch/pytorch",
    search_params="compute_cap<1200 num_gpus=1",
    autoscaler_instance="alpha"
)
fractal_ep.uv_pip_install(["Pillow"])

fractal_ep.ready()
