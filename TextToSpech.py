import json
import os
import sys
import time
import argparse
import subprocess
import shutil
import re
import tempfile

from indextts.infer_v2 import IndexTTS2

WAIT_RE = re.compile(
    r"<\s*wait\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?\s*>",
    flags=re.IGNORECASE
)
TARGET_RE = re.compile(
    r"<\s*(?:target|len|duration)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?\s*>",
    flags=re.IGNORECASE
)

# -------------------- helpers --------------------

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

def _parse_target_times_arg(s):
    """Parse --target_times '4,5,6.5' into list[float]."""
    if not s:
        return []
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            out.append(None)
    return out

def _read_target_times_file(path):
    if not path:
        return []
    if not os.path.exists(path):
        print(f"Warning: target_times_file not found: {path}")
        return []
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                vals.append(None)
                continue
            try:
                vals.append(float(ln))
            except Exception:
                vals.append(None)
    return vals

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
    if s is None:
        return ""
    s = str(s)
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- ")
    filtered = "".join(c for c in s if c in keep)
    name = "_".join(part for part in filtered.split() if part)
    name = name.strip("._-")[:max_len]
    return name or "utt"

def _extract_target_and_clean(text: str):
    """Remove <target X sec> tag and return (clean_text, target_seconds or None)."""
    target = None
    def repl(m):
        nonlocal target
        try:
            target = float(m.group(1))
        except Exception:
            target = None
        return " "
    clean = TARGET_RE.sub(repl, text)
    return clean.strip(), target

def _split_text_by_wait_tags(text):
    parts = []
    last = 0
    for m in WAIT_RE.finditer(text):
        if m.start() > last:
            segment = text[last:m.start()].strip()
            if segment:
                parts.append(("text", segment))
        secs = float(m.group(1))
        parts.append(("wait", secs))
        last = m.end()
    tail = text[last:].strip()
    if tail:
        parts.append(("text", tail))
    return parts if parts else [("text", text.strip())]

def _concat_wavs_with_silences(chunks, out_path):
    """chunks: [('wav', path), ('silence', secs), ...]"""
    try:
        from pydub import AudioSegment
    except Exception:
        raise RuntimeError("pydub required. `uv pip install pydub` and ensure ffmpeg in PATH.")
    final = None
    for kind, val in chunks:
        if kind == "wav":
            seg = AudioSegment.from_file(val)
        elif kind == "silence":
            ms = int(round(float(val) * 1000))
            seg = AudioSegment.silent(duration=ms)
        else:
            continue
        final = seg if final is None else (final + seg)
    if final is None:
        raise RuntimeError("Nothing to write; no audio produced.")
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    final.export(out_path, format="wav")

def _wav_duration_seconds(path):
    from pydub import AudioSegment
    return AudioSegment.from_file(path).duration_seconds

def _ffmpeg_atempo_chain(factor):
    """
    Builds an ffmpeg atempo chain approximating |factor|.
    atempo supports 0.5..2.0. We chain steps within that range.
    """
    steps = []
    remaining = factor
    # split into steps near 0.5..2
    while remaining < 0.5 or remaining > 2.0:
        if remaining < 1.0:
            steps.append(0.5)
            remaining /= 0.5
        else:
            steps.append(2.0)
            remaining /= 2.0
    steps.append(remaining)
    # merge near-1.0 steps
    filt = ",".join(f"atempo={s:.6f}" for s in steps)
    return filt

def _stretch_wav(in_path, out_path, factor):
    """
    factor >1.0 => speed up (shorter); <1.0 => slow down (longer)
    Uses ffmpeg atempo chain.
    """
    ffmpeg_bin = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found in PATH.")
    filt = _ffmpeg_atempo_chain(factor)
    cmd = [
        ffmpeg_bin, "-y", "-hide_banner",
        "-loglevel", "error",
        "-i", in_path,
        "-filter:a", filt,
        out_path
    ]
    subprocess.run(cmd, check=True)

def _infer_one_text_to_temp(tts: IndexTTS2, text: str, tmpdir: str, idx: int,
                            spk_audio_prompt: str,
                            emo_audio_prompt,
                            emo_alpha,
                            emo_vector,
                            use_emo_text,
                            emo_text,
                            use_random,
                            verbose):
    tmp_wav = os.path.join(tmpdir, f"seg_{idx:04d}.wav")
    tts.infer(
        spk_audio_prompt=spk_audio_prompt,
        text=text,
        output_path=tmp_wav,
        emo_audio_prompt=emo_audio_prompt,
        emo_alpha=emo_alpha,
        emo_vector=emo_vector,
        use_emo_text=use_emo_text,
        emo_text=emo_text,
        use_random=use_random,
        verbose=verbose
    )
    if not os.path.exists(tmp_wav):
        raise RuntimeError("Expected segment wav not found: " + tmp_wav)
    return tmp_wav

# -------------------- main --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Text->Speech using IndexTTS (wait tags + target duration)")
    parser.add_argument("--input", "-i", required=True, help="Text file, one line per utterance (supports 'TEXT || EMO_TEXT' and <wait X sec>, <target X sec>)")
    parser.add_argument("--spk_audio_prompt", "-p", required=True, help="Path to speaker prompt audio (voice.wav or .mp3)")
    parser.add_argument("--output_path", "-o", required=True,
                        help="Output dir or template. Dir => out/utt_<i>_<name>.wav. Template supports {i} and {name}.")
    parser.add_argument("--emo_alpha", "-a", type=float, default=0.65)
    parser.add_argument("--emo_vector", "-v", default="")
    parser.add_argument("--emo_audio_prompt", help="Path to emotional reference audio (optional)")
    parser.add_argument("--use_emo_text", action="store_true")
    parser.add_argument("--use_random", action="store_true")
    parser.add_argument("--emo_text", help="Global emotion text (overridden by per-line)")
    parser.add_argument("--line_delim", default="||", help="Delimiter between EMO_TEXT and per-line TEXT")
    parser.add_argument("--enable_wait_tags", action="store_true", help="Enable <wait X sec> inside text")
    parser.add_argument("--target_times", help="Comma list of target seconds per line, e.g. '4,5,6.5'")
    parser.add_argument("--target_times_file", help="File with one target seconds per line")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config", default="checkpoints/config.yaml")
    parser.add_argument("--model_dir", default="checkpoints")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_cuda_kernel", action="store_true")
    parser.add_argument("--use_deepspeed", action="store_true")
    args = parser.parse_args()

    tts = IndexTTS2(cfg_path=args.config, model_dir=args.model_dir,
                    use_fp16=args.use_fp16, use_cuda_kernel=args.use_cuda_kernel, use_deepspeed=args.use_deepspeed)

    if not os.path.exists(args.input):
        print("Input file not found:", args.input)
        sys.exit(1)

    args.spk_audio_prompt = _convert_mp3_to_wav(args.spk_audio_prompt) if args.spk_audio_prompt else args.spk_audio_prompt
    if args.emo_audio_prompt:
        args.emo_audio_prompt = _convert_mp3_to_wav(args.emo_audio_prompt)

    with open(args.input, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]

    # parse lines (TEXT [|| EMO_TEXT]); also extract inline <target ...>
    lines = []
    inline_targets = []
    delim = args.line_delim
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        # first split TEXT || EMO_TEXT (correct order)
        if delim and delim in line:
            emo_text_part ,text_part,  = [p.strip() for p in line.split(delim, 1)]
        else:
            emo_text_part ,text_part= line, None
        # extract <target ...> from text part
        clean_text, target_sec = _extract_target_and_clean(text_part)
        lines.append((clean_text, emo_text_part))
        inline_targets.append(target_sec)

    if not lines:
        print("No non-empty lines found in input.")
        sys.exit(1)

    # external targets
    targets_from_arg = _parse_target_times_arg(args.target_times) if args.target_times else []
    targets_from_file = _read_target_times_file(args.target_times_file) if args.target_times_file else []
    # merge priority: inline tag > --target_times > file > None
    target_seconds = []
    for i in range(len(lines)):
        t = inline_targets[i] if i < len(inline_targets) and inline_targets[i] is not None else None
        if t is None and i < len(targets_from_arg) and targets_from_arg[i] is not None:
            t = targets_from_arg[i]
        if t is None and i < len(targets_from_file) and targets_from_file[i] is not None:
            t = targets_from_file[i]
        target_seconds.append(t)

    emo_vec = _parse_vec(args.emo_vector)
    if emo_vec is not None:
        if len(emo_vec) < 8:
            emo_vec = emo_vec + [0.0] * (8 - len(emo_vec))
        emo_vec = [float(x) for x in emo_vec[:8]]

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
    if out_template.endswith(os.path.sep) or os.path.isdir(out_template):
        out_dir = out_template if out_template.endswith(os.path.sep) else out_template + os.path.sep
        os.makedirs(out_dir, exist_ok=True)
        def make_out(i, name):
            return os.path.join(out_dir, f"utt_{i}_{name}.wav")
    else:
        if "{i}" in out_template or "{name}" in out_template:
            def make_out(i, name):
                return out_template.format(i=i, name=name)
        else:
            base, ext = os.path.splitext(out_template)
            def make_out(i, name):
                return f"{base}_{i}{ext or '.wav'}"

    # -------------------- generation with timing control --------------------
    for idx, (text, per_line_emo_text) in enumerate(lines, start=1):
        name = _safe_filename(text) or f"line{idx}"
        out_path = make_out(idx, name)

        provided_this_line = provided_globally or (per_line_emo_text is not None)
        emo_text_to_pass = per_line_emo_text if per_line_emo_text else (args.emo_text if provided_globally and args.emo_text else None)
        use_emo_text_val = (args.use_emo_text if provided_globally else False) or (emo_text_to_pass is not None)

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

        tgt = target_seconds[idx-1] if idx-1 < len(target_seconds) else None

        try:
            tmpdir = tempfile.mkdtemp(prefix=f"tts_line_{idx:03d}_")
            chunks = []       # [('wav', path) or ('silence', secs)]
            voiced_wavs = []  # keep only wav paths for duration calc
            waits = []        # list of floats (secs) for waits in order

            if args.enable_wait_tags and WAIT_RE.search(text):
                print(f"[{idx}/{len(lines)}] Generating with <wait> tags -> {out_path} (target={tgt}s, emo_text={emo_text_to_pass})")
                seq = _split_text_by_wait_tags(text)

                seg_counter = 0
                for kind, val in seq:
                    if kind == "text" and val.strip():
                        seg_counter += 1
                        tmp_wav = _infer_one_text_to_temp(
                            tts, val, tmpdir, seg_counter,
                            spk_audio_prompt=args.spk_audio_prompt,
                            emo_audio_prompt=emo_audio_prompt_to_pass,
                            emo_alpha=emo_alpha_to_pass,
                            emo_vector=emo_vec_to_pass,
                            use_emo_text=use_emo_text_val,
                            emo_text=emo_text_to_pass,
                            use_random=use_random_to_pass,
                            verbose=args.verbose
                        )
                        chunks.append(("wav", tmp_wav))
                        voiced_wavs.append(tmp_wav)
                    elif kind == "wait":
                        waits.append(float(val))
                        chunks.append(("silence", float(val)))
            else:
                print(f"[{idx}/{len(lines)}] Generating -> {out_path} (target={tgt}s, emo_text={emo_text_to_pass})")
                # one voiced chunk only
                tmp_wav = _infer_one_text_to_temp(
                    tts, text, tmpdir, 1,
                    spk_audio_prompt=args.spk_audio_prompt,
                    emo_audio_prompt=emo_audio_prompt_to_pass,
                    emo_alpha=emo_alpha_to_pass,
                    emo_vector=emo_vec_to_pass,
                    use_emo_text=use_emo_text_val,
                    emo_text=emo_text_to_pass,
                    use_random=use_random_to_pass,
                    verbose=args.verbose
                )
                chunks.append(("wav", tmp_wav))
                voiced_wavs.append(tmp_wav)

            # --- timing control ---
            if tgt is not None:
                from pydub import AudioSegment  # for durations

                voiced_dur = sum(_wav_duration_seconds(p) for p in voiced_wavs)
                waits_dur = sum(waits) if waits else 0.0
                total = voiced_dur + waits_dur

                # utility to rebuild chunks with updated waits and (optionally) stretched voiced files
                def rebuild_and_export(stretched_voiced_paths, new_waits, path_out):
                    rebuilt = []
                    v_idx = 0
                    w_idx = 0
                    # walk original chunks order, replacing with new assets
                    for kind, val in chunks:
                        if kind == "wav":
                            rebuilt.append(("wav", stretched_voiced_paths[v_idx]))
                            v_idx += 1
                        else:
                            rebuilt.append(("silence", float(new_waits[w_idx])))
                            w_idx += 1
                    _concat_wavs_with_silences(rebuilt, path_out)

                # Case A: total > target → reduce waits first, then speed up voice if needed
                if total > tgt:
                    need_reduce = total - tgt
                    new_waits = list(waits)  # copy
                    if waits_dur > 0:
                        # proportional reduction of waits
                        reduce_from_waits = min(need_reduce, waits_dur)
                        scale = (waits_dur - reduce_from_waits) / waits_dur
                        new_waits = [w * scale for w in waits]
                        waits_after = sum(new_waits)
                        remain_delta = total - (voiced_dur + waits_after)
                    else:
                        waits_after = 0.0
                        new_waits = []
                        remain_delta = need_reduce

                    # if still too long, speed up voiced by factor f > 1
                    if remain_delta > 1e-3 and voiced_dur > 0:
                        desired_voiced = voiced_dur - remain_delta
                        desired_voiced = max(desired_voiced, 0.05)  # floor
                        speed_factor = voiced_dur / desired_voiced  # >1
                        stretched = []
                        for i, vp in enumerate(voiced_wavs, 1):
                            outp = os.path.join(tmpdir, f"seg_stretch_{i:04d}.wav")
                            _stretch_wav(vp, outp, speed_factor)
                            stretched.append(outp)
                        rebuild_and_export(stretched, new_waits, out_path)
                    else:
                        # waits reduction sufficed
                        rebuild_and_export(voiced_wavs, new_waits, out_path)

                # Case B: total < target → increase waits if present; else slow voice
                elif total < tgt - 1e-3:
                    need_increase = tgt - total
                    if waits_dur > 0:
                        # proportional increase of waits
                        new_waits = [w + (w / waits_dur) * need_increase for w in waits]
                        rebuild_and_export(voiced_wavs, new_waits, out_path)
                    else:
                        # slow down voiced
                        desired_voiced = voiced_dur + need_increase
                        slow_factor = voiced_dur / desired_voiced  # <1
                        slowed = []
                        for i, vp in enumerate(voiced_wavs, 1):
                            outp = os.path.join(tmpdir, f"seg_stretch_{i:04d}.wav")
                            _stretch_wav(vp, outp, slow_factor)
                            slowed.append(outp)
                        # still no waits → use zeros
                        rebuild_and_export(slowed, [], out_path)
                else:
                    # already within ~1ms
                    _concat_wavs_with_silences(chunks, out_path)
            else:
                # no target: normal concat
                _concat_wavs_with_silences(chunks, out_path)

        except Exception as e:
            print(f"Failed to generate line {idx}: {e}")
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
