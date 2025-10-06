import json
import os
import sys
import threading
import time
import argparse

from indextts.infer_v2 import IndexTTS2


def _parse_vec(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        if s.startswith("[") and s.endswith("]"):
            return [float(x) for x in json.loads(s)]
        return [float(x) for x in s.split(",")]
    except Exception:
        return None


def _safe_filename(s):
    keepchars = (" ", ".", "_", "-")
    return "".join(c for c in s if c.isalnum() or c in keepchars).strip().replace(" ", "_")[:64]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Text->Speech using IndexTTS")
    parser.add_argument("--input", "-i", required=True, help="Text file, one line per utterance")
    parser.add_argument("--spk_audio_prompt", "-p", required=True, help="Path to speaker prompt audio (voice.wav)")
    parser.add_argument("--output_path", "-o", required=True,
                        help="Output path or directory. If directory, files saved as <output_path>/utt_<i>.wav. "
                             "You can use template with {i} and {name}, e.g. out/utt_{i}_{name}.wav")
    parser.add_argument("--emo_alpha", "-a", type=float, default=0.65, help="Emotion alpha (emo_alpha)")
    parser.add_argument("--emo_vector", "-v", default="", help="Emotion vector, e.g. '0,0,0,0,0,0,0.45,0' or JSON list")
    parser.add_argument("--use_random", action="store_true", help="Use emo random flag")
    parser.add_argument("--verbose", action="store_true", help="Verbose infer")
    parser.add_argument("--config", default="checkpoints/config.yaml", help="Path to config.yaml")
    parser.add_argument("--model_dir", default="checkpoints", help="Model directory")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_cuda_kernel", action="store_true")
    parser.add_argument("--use_deepspeed", action="store_true")
    args = parser.parse_args()

    # instantiate tts
    tts = IndexTTS2(cfg_path=args.config, model_dir=args.model_dir,
                    use_fp16=args.use_fp16, use_cuda_kernel=args.use_cuda_kernel, use_deepspeed=args.use_deepspeed)

    # read input lines
    if not os.path.exists(args.input):
        print("Input file not found:", args.input)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        print("No non-empty lines found in input.")
        sys.exit(1)

    emo_vec = _parse_vec(args.emo_vector)
    # if provided vector shorter than 8, pad; if None, leave None
    if emo_vec is not None:
        if len(emo_vec) < 8:
            emo_vec = emo_vec + [0.0] * (8 - len(emo_vec))
        emo_vec = [float(x) for x in emo_vec[:8]]

    out_template = args.output_path
    # if output path is a directory or endswith sep -> create dir and use pattern
    if out_template.endswith(os.path.sep) or os.path.isdir(out_template):
        out_dir = out_template if out_template.endswith(os.path.sep) else out_template + os.path.sep
        os.makedirs(out_dir, exist_ok=True)
        def make_out(i, name):
            return os.path.join(out_dir, f"utt_{i}_{name}.wav")
    else:
        # if contains {i} or {name}, use formatting, else append _{i}.wav
        if "{i}" in out_template or "{name}" in out_template:
            def make_out(i, name):
                return out_template.format(i=i, name=name)
        else:
            base, ext = os.path.splitext(out_template)
            def make_out(i, name):
                return f"{base}_{i}{ext or '.wav'}"

    # generate
    for idx, line in enumerate(lines, start=1):
        name = _safe_filename(line) or f"line{idx}"
        out_path = make_out(idx, name)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        try:
            print(f"[{idx}/{len(lines)}] Generating -> {out_path}")
            tts.infer(spk_audio_prompt=args.spk_audio_prompt,
                      text=line,
                      output_path=out_path,
                      emo_audio_prompt=None,
                      emo_alpha=args.emo_alpha,
                      emo_vector=emo_vec,
                      use_random=args.use_random,
                      verbose=args.verbose)
        except Exception as e:
            print(f"Failed to generate line {idx}: {e}")
            continue