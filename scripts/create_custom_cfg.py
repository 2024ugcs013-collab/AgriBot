"""
Potato Leaf Disease Detection with YOLOv4
Stage 6: Configuration File (yolov4-custom-potato.cfg)

Edits:
- batch=64, subdivisions=16, width=416, height=416
- learning_rate=0.001
- max_batches=6000, steps=4800,5400
- classes=3 in all 3 [yolo] layers
- filters=24 in the convolutional layer immediately preceding each [yolo] layer
- Comprehensive parser assertions and documentation of the [net] hyperparameters.
"""

from pathlib import Path
import re
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = PROJECT_ROOT / "cfg"
TEMPLATE_CFG = CFG_DIR / "yolov4-custom.cfg"
TARGET_CFG = CFG_DIR / "yolov4-custom-potato.cfg"

def build_yolov4_cfg():
    print("=" * 60)
    print("STAGE 6: YOLOv4 CUSTOM CONFIGURATION")
    print("=" * 60)

    if not TEMPLATE_CFG.exists():
        raise FileNotFoundError(f"Missing template cfg: {TEMPLATE_CFG}")

    # Read original lines
    lines = TEMPLATE_CFG.read_text(encoding="utf-8").splitlines()

    changes = []

    # ---------------------------------------------------------
    # Parse blocks: list of dicts with 'name', 'lines'
    # ---------------------------------------------------------
    blocks = []
    current_block = {"name": None, "lines": []}

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current_block["name"] is not None or current_block["lines"]:
                blocks.append(current_block)
            current_block = {"name": stripped, "lines": [line]}
        else:
            current_block["lines"].append(line)
    if current_block["name"] is not None or current_block["lines"]:
        blocks.append(current_block)

    # ---------------------------------------------------------
    # Modify [net] section
    # ---------------------------------------------------------
    net_block = blocks[0]
    assert net_block["name"] == "[net]", "First block must be [net]"

    net_replacements = {
        "batch": "64",
        "subdivisions": "16",
        "width": "416",
        "height": "416",
        "learning_rate": "0.001",
        "max_batches": "6000",
        "steps": "4800,5400"
    }

    new_net_lines = []
    for line in net_block["lines"]:
        matched = False
        for key, new_val in net_replacements.items():
            # Match active or commented key
            pattern = rf"^\s*#?\s*{key}\s*=\s*.*"
            if re.match(pattern, line.strip()):
                old_val = line.strip()
                new_line = f"{key}={new_val}"
                new_net_lines.append(new_line)
                changes.append({"section": "[net]", "parameter": key, "old": old_val, "new": new_line})
                matched = True
                break
        if not matched:
            new_net_lines.append(line)

    net_block["lines"] = new_net_lines

    # ---------------------------------------------------------
    # Modify [yolo] sections and their preceding [convolutional]
    # ---------------------------------------------------------
    yolo_count = 0
    for idx, block in enumerate(blocks):
        if block["name"] == "[yolo]":
            yolo_count += 1
            # 1. Update classes=3 in [yolo] block
            new_yolo_lines = []
            for l in block["lines"]:
                if re.match(r"^\s*classes\s*=\s*.*", l.strip()):
                    changes.append({"section": f"[yolo] #{yolo_count}", "parameter": "classes", "old": l.strip(), "new": "classes=3"})
                    new_yolo_lines.append("classes=3")
                else:
                    new_yolo_lines.append(l)
            block["lines"] = new_yolo_lines

            # 2. Update filters=24 in the preceding [convolutional] block
            prev_idx = idx - 1
            while prev_idx >= 0 and blocks[prev_idx]["name"] != "[convolutional]":
                prev_idx -= 1
            assert prev_idx >= 0, f"No preceding [convolutional] found for [yolo] #{yolo_count}"
            
            conv_block = blocks[prev_idx]
            new_conv_lines = []
            for l in conv_block["lines"]:
                if re.match(r"^\s*filters\s*=\s*.*", l.strip()):
                    changes.append({"section": f"[convolutional] before [yolo] #{yolo_count}", "parameter": "filters", "old": l.strip(), "new": "filters=24"})
                    new_conv_lines.append("filters=24")
                else:
                    new_conv_lines.append(l)
            conv_block["lines"] = new_conv_lines

    assert yolo_count == 3, f"Expected 3 [yolo] sections, found {yolo_count}"

    # Reconstruct modified file
    output_lines = []
    for block in blocks:
        output_lines.extend(block["lines"])

    TARGET_CFG.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"Generated customized configuration: {TARGET_CFG}")

    # ---------------------------------------------------------
    # Verification Assertions on Saved File
    # ---------------------------------------------------------
    print("\nRunning verification assertions on generated cfg...")
    content = TARGET_CFG.read_text(encoding="utf-8")
    
    # 1. Check [yolo] layers
    yolo_classes = re.findall(r"\[yolo\][\s\S]*?classes\s*=\s*(\d+)", content)
    assert len(yolo_classes) == 3, f"Expected 3 yolo classes matches, got {len(yolo_classes)}"
    assert all(c == "3" for c in yolo_classes), f"YOLO classes mismatch: {yolo_classes}"

    # 2. Check filters in [convolutional] immediately preceding each [yolo]
    # Parse blocks again from generated file to verify strict structure
    gen_lines = TARGET_CFG.read_text(encoding="utf-8").splitlines()
    gen_blocks = []
    curr_b = []
    for l in gen_lines:
        s = l.strip()
        if s.startswith("[") and s.endswith("]"):
            if curr_b:
                gen_blocks.append(curr_b)
            curr_b = [s]
        else:
            curr_b.append(l)
    if curr_b:
        gen_blocks.append(curr_b)

    verified_yolo_layers = 0
    for idx, b in enumerate(gen_blocks):
        if b[0] == "[yolo]":
            verified_yolo_layers += 1
            # Check classes=3
            yolo_classes_line = [l.strip() for l in b if l.strip().startswith("classes")]
            assert yolo_classes_line == ["classes=3"], f"Expected classes=3, got {yolo_classes_line}"

            # Check preceding conv layer has filters=24
            prev_b = gen_blocks[idx - 1]
            assert prev_b[0] == "[convolutional]", f"Expected [convolutional] before [yolo], got {prev_b[0]}"
            conv_filters_line = [l.strip() for l in prev_b if l.strip().startswith("filters")]
            assert conv_filters_line == ["filters=24"], f"Expected filters=24, got {conv_filters_line}"

    assert verified_yolo_layers == 3, f"Expected 3 verified yolo layers, got {verified_yolo_layers}"

    # 3. Check [net] parameters
    assert re.search(r"^batch\s*=\s*64", content, re.MULTILINE), "batch=64 assertion failed"
    assert re.search(r"^subdivisions\s*=\s*16", content, re.MULTILINE), "subdivisions=16 assertion failed"
    assert re.search(r"^width\s*=\s*416", content, re.MULTILINE), "width=416 assertion failed"
    assert re.search(r"^height\s*=\s*416", content, re.MULTILINE), "height=416 assertion failed"
    assert re.search(r"^learning_rate\s*=\s*0\.001", content, re.MULTILINE), "learning_rate=0.001 assertion failed"
    assert re.search(r"^max_batches\s*=\s*6000", content, re.MULTILINE), "max_batches=6000 assertion failed"
    assert re.search(r"^steps\s*=\s*4800,5400", content, re.MULTILINE), "steps=4800,5400 assertion failed"

    print("ALL CONFIG ASSERTIONS PASSED!")

    # ---------------------------------------------------------
    # Changes Summary Table
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'SECTION':<35} | {'PARAMETER':<12} | {'OLD VALUE':<16} | {'NEW VALUE':<16}")
    print("-" * 85)
    for c in changes:
        print(f"{c['section']:<35} | {c['parameter']:<12} | {c['old']:<16} | {c['new']:<16}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Print [net] Section
    # ---------------------------------------------------------
    print("\n--- [net] SECTION IN OUTPUT CFG ---")
    for line in net_block["lines"][:25]:
        print(line)

    print("\n" + "=" * 60)
    print("STAGE 6 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    build_yolov4_cfg()
