import json
import os
import sys
import threading
import time
import argparse
import subprocess
import shutil

from indextts.infer_v2 import IndexTTS2

"""
example usage:
python TextToSpech.py \
  --input text.txt \
  --spk_audio_prompt voice.wav \
  --output_path outputs/tts/ \
  --emo_alpha 0.65 \
  --emo_vector "0,0,0,0,0,0,0.45,0"

Input file formats (each non-empty line processed in order):
- Simple text per line:
    Hello world.
    This is a test.

- Provide per-line emotion text using delimiter (default "||"):
    Hello world. || happy and excited
    这是中文示例。 || calm and melancholic

Notes:
- Delimiter can be changed with --line_delim.
- emo_text (after the delimiter) will be passed to infer via emo_text param.
- emo_vector / emo_audio_prompt / emo_alpha still apply globally via CLI flags.
"""



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

# --- new helper: convert mp3 -> wav using ffmpeg or pydub ---
def _convert_mp3_to_wav(src_path):
    if not src_path:
        return None
    src_path = str(src_path)
    if not src_path.lower().endswith(".mp3"):
        return src_path
    if not os.path.exists(src_path):
        return src_path
    out_path = os.path.splitext(src_path)[0] + ".wav"
    if os.path.exists(out_path):
        return out_path
    ffmpeg_bin = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if ffmpeg_bin:
        try:
            cmd = [ffmpeg_bin, "-y", "-i", src_path, out_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out_path):
                print(f"Converted mp3 -> wav: {src_path} -> {out_path}")
                return out_path
        except Exception as e:
            print("ffmpeg conversion failed:", e)
    # fallback to pydub if installed
    try:
        from pydub import AudioSegment
        AudioSegment.from_file(src_path).export(out_path, format="wav")
        if os.path.exists(out_path):
            print(f"Converted mp3 -> wav (pydub): {src_path} -> {out_path}")
            return out_path
    except Exception:
        pass
    print("Warning: could not convert mp3 to wav, using original file:", src_path)
    return src_path


def _safe_filename(s: str, max_len: int = 64) -> str:
    """
    Produce a filesystem-safe short name from arbitrary text.
    - keeps alphanumeric and . _ - characters
    - replaces spaces with underscores
    - strips leading/trailing dots/underscores
    - truncates to max_len
    """
    if s is None:
        return ""
    s = str(s)
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- ")
    filtered = "".join(c for c in s if c in keep)
    # normalize spaces, replace with underscores
    name = "_".join(part for part in filtered.split() if part)
    # strip leading/trailing separators and limit length
    name = name.strip("._-")[:max_len]
    return name or "utt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Text->Speech using IndexTTS")
    parser.add_argument("--input", "-i", required=True, help="Text file, one line per utterance (optionally 'TEXT || EMO_TEXT')")
    parser.add_argument("--spk_audio_prompt", "-p", required=True, help="Path to speaker prompt audio (voice.wav)")
    parser.add_argument("--output_path", "-o", required=True,
                        help="Output path or directory. If directory, files saved as <output_path>/utt_<i>_<name>.wav. "
                             "You can use template with {i} and {name}, e.g. out/utt_{i}_{name}.wav")
    parser.add_argument("--emo_alpha", "-a", type=float, default=0.65, help="Emotion alpha (emo_alpha)")
    parser.add_argument("--emo_vector", "-v", default="", help="Emotion vector, e.g. '0,0,0,0,0,0,0.45,0' or JSON list")
    parser.add_argument("--emo_audio_prompt", help="Path to separate emotional reference audio (optional)")
    parser.add_argument("--use_emo_text", action="store_true", help="Use text-to-emotion mode (convert text to emotion vectors)")
    parser.add_argument("--use_random", action="store_true", help="Use emo random flag")
    parser.add_argument("--emo_text", help="Explicit emotion text to use for ALL lines (overridden by per-line emo_text if present)")
    parser.add_argument("--line_delim", default="||", help="Delimiter between text and per-line emo_text in input (default '||')")
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

    # convert reference audios from mp3 to wav if needed
    args.spk_audio_prompt = _convert_mp3_to_wav(args.spk_audio_prompt) if args.spk_audio_prompt else args.spk_audio_prompt
    if args.emo_audio_prompt:
        args.emo_audio_prompt = _convert_mp3_to_wav(args.emo_audio_prompt)

    with open(args.input, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]

    # parse lines into (text, per_line_emo_text)
    lines = []
    delim = args.line_delim
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        if delim and delim in line:
            parts = line.split(delim, 1)
            text_part = parts[0].strip()
            emo_text_part = parts[1].strip()
            lines.append((text_part, emo_text_part))
        else:
            lines.append((line, None))

    if not lines:
        print("No non-empty lines found in input.")
        sys.exit(1)

    emo_vec = _parse_vec(args.emo_vector)
    # if provided vector shorter than 8, pad; if None, leave None
    if emo_vec is not None:
        if len(emo_vec) < 8:
            emo_vec = emo_vec + [0.0] * (8 - len(emo_vec))
        emo_vec = [float(x) for x in emo_vec[:8]]

    # determine if user provided any global emotion-related args
    provided_globally = False
    try:
        provided_globally = bool(
            (args.emo_audio_prompt and str(args.emo_audio_prompt).strip()) or
            (args.emo_vector and str(args.emo_vector).strip()) or
            args.use_emo_text or
            (args.emo_text and str(args.emo_text).strip()) or
            args.use_random
        )
    except Exception:
        provided_globally = False

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
    for idx, (text, per_line_emo_text) in enumerate(lines, start=1):
        name = _safe_filename(text) or f"line{idx}"
        out_path = make_out(idx, name)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # decide whether to apply emotion conditioning for this line:
        # - apply if any global emotion args were provided, or if this line has per-line emo_text
        provided_this_line = provided_globally or (per_line_emo_text is not None)

        # emo_text preference: per-line emo_text overrides global emo_text
        emo_text_to_pass = per_line_emo_text if per_line_emo_text else (args.emo_text if provided_globally and args.emo_text else None)
        # decide use_emo_text flag for this call
        use_emo_text_val = (args.use_emo_text if provided_globally else False) or (emo_text_to_pass is not None)

        # prepare per-call emotion parameters (clear them if not provided for this line)
        if provided_this_line:
            emo_audio_prompt_to_pass = args.emo_audio_prompt if args.emo_audio_prompt else None
            emo_vec_to_pass = emo_vec
            emo_alpha_to_pass = args.emo_alpha
            use_random_to_pass = args.use_random
        else:
            emo_audio_prompt_to_pass = None
            emo_vec_to_pass = None
            emo_alpha_to_pass = 0.0
            use_random_to_pass = False
        try:
            print(f"[{idx}/{len(lines)}] Generating -> {out_path}  (emo_text: {emo_text_to_pass})")
            tts.infer(spk_audio_prompt=args.spk_audio_prompt,
                      text=text,
                      output_path=out_path,
                      emo_audio_prompt=emo_audio_prompt_to_pass,
                      emo_alpha=emo_alpha_to_pass,
                      emo_vector=emo_vec_to_pass,
                      use_emo_text=use_emo_text_val,
                      emo_text=emo_text_to_pass,
                      use_random=use_random_to_pass,
                      verbose=args.verbose)
        except Exception as e:
             print(f"Failed to generate line {idx}: {e}")
             continue