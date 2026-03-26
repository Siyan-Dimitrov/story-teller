"""Debug script for Replicate Flux API integration.

Run: python debug_replicate.py
Each step is isolated so you can see exactly where the pipeline breaks.
"""

import os
import sys
import time

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"

def step(num, title):
    print(f"\n{'='*60}")
    print(f"  Step {num}: {title}")
    print(f"{'='*60}")


# ── Step 1: Check env var ────────────────────────────────────
step(1, "REPLICATE_API_TOKEN environment variable")

token = os.environ.get("REPLICATE_API_TOKEN", "")
if not token:
    print(f"{FAIL} REPLICATE_API_TOKEN is not set in your environment.")
    print()
    print("  To set it permanently in Windows:")
    print("    1. Win+R -> sysdm.cpl -> Advanced -> Environment Variables")
    print("    2. Under 'User variables', click New")
    print("       Name:  REPLICATE_API_TOKEN")
    print("       Value: r8_YourTokenHere")
    print("    3. Click OK, then restart your terminal / IDE")
    print()
    print("  Or set it temporarily for this session:")
    print("    set REPLICATE_API_TOKEN=r8_YourTokenHere")
    sys.exit(1)

# Mask token for display
masked = token[:5] + "..." + token[-4:] if len(token) > 12 else "***"
print(f"{PASS} Token found: {masked}  (length={len(token)})")

if not token.startswith("r8_"):
    print(f"{WARN} Token doesn't start with 'r8_' — Replicate tokens usually do.")
    print(f"      Double-check your token at: https://replicate.com/account/api-tokens")
else:
    print(f"{PASS} Token format looks correct (r8_ prefix)")


# ── Step 2: Import replicate library ─────────────────────────
step(2, "Import replicate Python library")

try:
    import replicate
    print(f"{PASS} replicate library imported (version: {getattr(replicate, '__version__', 'unknown')})")
except ImportError as e:
    print(f"{FAIL} Cannot import replicate: {e}")
    print("      Install it with: pip install replicate")
    sys.exit(1)


# ── Step 3: Test API authentication ─────────────────────────
step(3, "API authentication check")

try:
    client = replicate.Client(api_token=token)
    # Use a direct model lookup as a lightweight auth check
    test_model = client.models.get("black-forest-labs", "flux-schnell")
    print(f"{PASS} Authenticated successfully (verified model: {test_model.owner}/{test_model.name})")
except Exception as e:
    err_str = str(e).lower()
    if "401" in err_str or "unauthorized" in err_str or "authentication" in err_str:
        print(f"{FAIL} Authentication failed: {e}")
        print("      Check your token is valid and not expired.")
        print("      Get a new token: https://replicate.com/account/api-tokens")
    else:
        print(f"{FAIL} API check failed: {e}")
    sys.exit(1)


# ── Step 4: Check Flux model availability ────────────────────
step(4, "Flux model availability")

model_id = os.environ.get("REPLICATE_MODEL", "black-forest-labs/flux-schnell")
print(f"{INFO} Target model: {model_id}")

try:
    owner, name = model_id.split("/")[:2]
    model = client.models.get(owner, name)
    print(f"{PASS} Model found: {model.owner}/{model.name}")
    print(f"      Description: {(model.description or '')[:100]}")
    if model.default_example:
        print(f"      Has default example: yes")
except Exception as e:
    print(f"{FAIL} Cannot find model '{model_id}': {e}")
    print("      Available Flux models:")
    print("        black-forest-labs/flux-schnell  (fast, cheaper)")
    print("        black-forest-labs/flux-dev      (higher quality)")
    print("        black-forest-labs/flux-1.1-pro  (best quality)")
    sys.exit(1)


# ── Step 5: Run a test generation ────────────────────────────
step(5, "Test image generation")

test_prompt = "a dark fairy tale castle on a misty hill, gothic storybook illustration, moody lighting"
inp = {
    "prompt": test_prompt,
    "width": 1024,
    "height": 1024,
    "num_outputs": 1,
    "output_format": "png",
}

if "schnell" in model_id:
    inp["num_inference_steps"] = 4
    print(f"{INFO} Using schnell — 4 inference steps (fast mode)")
else:
    inp["num_inference_steps"] = 28
    print(f"{INFO} Using {model_id} — 28 inference steps")

print(f"{INFO} Prompt: \"{test_prompt[:80]}...\"")
print(f"{INFO} Sending request to Replicate...")

t0 = time.time()
try:
    output = replicate.run(model_id, input=inp)
    elapsed = time.time() - t0
    print(f"{PASS} Generation completed in {elapsed:.1f}s")
except Exception as e:
    elapsed = time.time() - t0
    print(f"{FAIL} Generation failed after {elapsed:.1f}s: {e}")
    print()
    err_str = str(e).lower()
    if "nsfw" in err_str or "safety" in err_str:
        print("      The prompt was flagged by Replicate's safety filter.")
        print("      Try a different prompt.")
    elif "billing" in err_str or "payment" in err_str or "limit" in err_str:
        print("      Billing issue — check your Replicate account has credits.")
        print("      https://replicate.com/account/billing")
    elif "timeout" in err_str:
        print("      Request timed out. The model may be cold-starting.")
        print("      Try again in 30 seconds.")
    else:
        print(f"      Full error: {e}")
    sys.exit(1)


# ── Step 6: Validate output ─────────────────────────────────
step(6, "Validate output")

if isinstance(output, list):
    print(f"{PASS} Got list output with {len(output)} item(s)")
    image_ref = output[0]
else:
    print(f"{WARN} Output is not a list (type={type(output).__name__}), using directly")
    image_ref = output

image_url = str(image_ref)
print(f"{INFO} Image URL/ref: {image_url[:120]}...")


# ── Step 7: Download the image ───────────────────────────────
step(7, "Download image")

try:
    import httpx
    with httpx.Client(timeout=60) as http:
        resp = http.get(image_url)
        resp.raise_for_status()
        image_bytes = resp.content
    print(f"{PASS} Downloaded {len(image_bytes):,} bytes")
except ImportError:
    # Fall back to urllib
    import urllib.request
    req = urllib.request.Request(image_url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        image_bytes = resp.read()
    print(f"{PASS} Downloaded {len(image_bytes):,} bytes (via urllib)")
except Exception as e:
    print(f"{FAIL} Download failed: {e}")
    sys.exit(1)


# ── Step 8: Save and verify ─────────────────────────────────
step(8, "Save and verify image file")

output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "debug_replicate_test.png")

with open(output_path, "wb") as f:
    f.write(image_bytes)
print(f"{PASS} Saved to: {output_path}")
print(f"      File size: {os.path.getsize(output_path):,} bytes")

# Basic PNG validation (PNG magic bytes)
with open(output_path, "rb") as f:
    header = f.read(8)

if header[:4] == b'\x89PNG':
    print(f"{PASS} Valid PNG file")
else:
    print(f"{WARN} File doesn't have PNG header (got: {header[:4].hex()})")
    print(f"      The output_format may not be PNG, but the image may still be valid.")

# Try to get dimensions via PIL if available
try:
    from PIL import Image
    img = Image.open(output_path)
    print(f"{PASS} Image dimensions: {img.size[0]}x{img.size[1]}, mode={img.mode}")
    img.close()
except ImportError:
    print(f"{INFO} PIL not installed — skipping dimension check")
except Exception as e:
    print(f"{WARN} PIL could not open image: {e}")


# ── Summary ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ALL STEPS PASSED")
print(f"{'='*60}")
print(f"  Replicate Flux API is working correctly.")
print(f"  Model:  {model_id}")
print(f"  Speed:  {elapsed:.1f}s")
print(f"  Output: {output_path}")
print(f"{'='*60}\n")
