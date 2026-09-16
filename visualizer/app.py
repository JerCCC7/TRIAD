#!/usr/bin/env python3
"""Local, single-operator TRIAD ERP rotation demo."""
import argparse
import base64
import io
import json
import mimetypes
import sys
import threading
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize, InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(ROOT))
from triad.checkpoint import choose_device, load_checkpoint
from triad.data import EXTENSIONS, png_roundtrip, save_image
from triad.rotation import quaternion_matrix, rotate_panorama


class State:
    def __init__(self, args):
        self.args = args
        self.device = choose_device(args.device)
        self.lock = threading.Lock()
        self.model = None
        self.image = self.watermarked = self.bits = self.session = None
        self.rotation_index = self.embed_index = 0
        self.files = sorted(p.resolve() for p in args.data_dir.rglob("*") if p.suffix.lower() in EXTENSIONS)
        checkpoint = args.checkpoint
        if checkpoint.is_dir():
            self.config = json.loads((checkpoint / "config.json").read_text())
        else:
            c = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.config = c.get("model_config", c.get("args", {}))
        self.height = int(self.config.get("resolution", 256))

    def load_image(self, image, name):
        image = resize(image.convert("RGB"), [self.height, 2 * self.height], InterpolationMode.BILINEAR)
        self.image = (to_tensor(image).unsqueeze(0) * 2 - 1).to(self.device)
        self.watermarked = self.bits = None
        self.session = self.args.output_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
        self.session.mkdir(parents=True)
        self.rotation_index = self.embed_index = 0
        save_image(self.image, self.session / "original.png")
        return dict(image_name=name, session=self.session.name, original_url=self.url("original.png"))

    def ensure_model(self):
        if self.model is None:
            self.model, self.config = load_checkpoint(self.args.checkpoint, self.device)

    def url(self, name):
        return "/outputs/" + self.session.relative_to(self.args.output_dir).as_posix() + "/" + name

    def scores(self, image):
        prediction = self.model.extract(image) > 0
        errors = int((prediction != self.bits.bool()).sum())
        return dict(ber=errors / self.bits.numel(), accuracy=1 - errors / self.bits.numel(),
                    errors=errors, num_bits=self.bits.numel(),
                    true_bits=self.bits[0].int().cpu().tolist(), pred_bits=prediction[0].int().cpu().tolist())

    @torch.inference_mode()
    def embed(self, seed):
        if self.image is None:
            raise ValueError("Load an ERP image first")
        self.ensure_model()
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.bits = torch.randint(2, (1, self.config["latent_dim"]), generator=generator, device=self.device).float()
        self.watermarked = png_roundtrip(self.model.embed(self.image, self.bits))
        self.embed_index += 1
        prefix = f"embed_{self.embed_index:03d}"
        name = prefix + ".png"
        save_image(self.watermarked, self.session / name)
        scores = self.scores(self.watermarked)
        (self.session / (prefix + ".json")).write_text(json.dumps(dict(seed=seed, model=self.config, **scores), indent=2))
        return dict(watermarked_url=self.url(name), clean_ber=scores["ber"],
                    clean_accuracy=scores["accuracy"], true_bits=scores["true_bits"], session=self.session.name)

    @torch.inference_mode()
    def rotate(self, values):
        if self.watermarked is None:
            raise ValueError("Embed a watermark first")
        matrix = quaternion_matrix(values, self.device)
        rotated = png_roundtrip(rotate_panorama(self.watermarked, matrix))
        self.rotation_index += 1
        prefix = f"embed_{self.embed_index:03d}_rotation_{self.rotation_index:03d}"
        save_image(rotated, self.session / (prefix + ".png"))
        # Decode the same 8-bit pixel values that were saved to disk.
        scores = self.scores(rotated)
        report = dict(rotation_quaternion_xyzw=values, rotation_matrix=matrix[0].cpu().tolist(),
                      sampler="spherical", quantization="uint8_png", **scores)
        (self.session / (prefix + ".json")).write_text(json.dumps(report, indent=2))
        return dict(rotated_url=self.url(prefix + ".png"), extraction_json=self.url(prefix + ".json"), **scores)


class Handler(BaseHTTPRequestHandler):
    state: State

    def send(self, data, status=200, mime="application/json; charset=utf-8"):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def file(self, root, name):
        path = (root / unquote(name)).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            self.send({"error": "Not found"}, 404)
            return
        mime = "text/javascript" if path.suffix == ".js" else mimetypes.guess_type(path.name)[0]
        self.send(path.read_bytes(), mime=mime or "application/octet-stream")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.file(STATIC, "index.html")
        elif path.startswith("/static/"):
            self.file(STATIC, path[len("/static/"):])
        elif path.startswith("/outputs/"):
            self.file(self.state.args.output_dir, path[len("/outputs/"):])
        elif path == "/api/config":
            self.send(dict(device=self.state.device, height=self.state.height, width=2 * self.state.height,
                           model_info=self.state.config, model_loaded=self.state.model is not None))
        elif path == "/api/images":
            self.send(dict(images=[dict(name=str(p.relative_to(self.state.args.data_dir)), path=str(i))
                                   for i, p in enumerate(self.state.files)]))
        else:
            self.send({"error": "Not found"}, 404)

    def do_POST(self):
        if self.headers.get("Origin") and urlparse(self.headers["Origin"]).netloc != self.headers.get("Host"):
            self.send({"error": "Cross-origin request rejected"}, 403)
            return
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.send({"error": "JSON required"}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32 * 1024 * 1024:
                self.send({"error": "Upload must be smaller than 32 MB"}, 413)
                return
            payload = json.loads(self.rfile.read(length))
            with self.state.lock:
                if self.path == "/api/load_path":
                    index = int(payload["path"])
                    if not 0 <= index < len(self.state.files):
                        raise ValueError("Invalid image index")
                    path = self.state.files[index]
                    with Image.open(path) as image:
                        result = self.state.load_image(image, path.name)
                elif self.path == "/api/load_upload":
                    raw = base64.b64decode(payload["data_url"].split(",", 1)[1], validate=True)
                    with Image.open(io.BytesIO(raw)) as image:
                        result = self.state.load_image(image, payload.get("name", "upload"))
                elif self.path == "/api/embed":
                    result = self.state.embed(payload.get("seed", 42))
                elif self.path == "/api/rotate_extract":
                    result = self.state.rotate(payload["rotation_quaternion"])
                else:
                    self.send({"error": "Not found"}, 404)
                    return
            self.send(result)
        except (ValueError, KeyError, IndexError) as exc:
            self.send({"error": str(exc)}, 400)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.send({"error": str(exc)}, 500)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/triad-32bit")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/test")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/visualizer")
    p.add_argument("--device", default="auto")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7861)
    args = p.parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    Handler.state = State(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"TRIAD visualizer: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
