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

# ---------- token regexes ----------
TARGET_RE = re.compile(
    r"<\s*(?:target|len|duration)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?\s*>",
    flags=re.IGNORECASE
)
# Unified tokenizer for wait + emo + emo-clear
TOKEN_RE = re.compile(
    r"(?P<wait><\s*wait\s+(?P<sec>\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?\s*>)"
    r"|(?P<emo><\s*emo(?::|\s+)(?P<emo_text>[^>]+)>)"
    r"|(?P<clear><\s*emo[-_\s]*clear\s*>)",
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

def _normalize_emo_text(raw: str) -> str:
    # Collapse commas/whitespace to a single-space phrase
    tokens = re.split(r"[\s,]+", (raw or "").strip())
    tokens = [t for t in tokens if t]
    return " ".join(tokens) if tokens else ""

def _tokenize_with_emo_and_wait(text: str, initial_emo: str = None):
    """
    Return sequence preserving order:
      ('text', segment_text, emo_text_or_None)
      ('wait', seconds)
    Emotion is a sticky state set by <emo ...> and cleared by <emo-clear>.
    """
    events = []
    last = 0
    # raw events w/o emo applied
    raw_events = []
    for m in TOKEN_RE.finditer(text):
        if m.start() > last:
            pre = text[last:m.start()].strip()
            if pre:
                raw_events.append(("text", pre))
        if m.group("wait"):
            secs = float(m.group("sec"))
            raw_events.append(("wait", secs))
        elif m.group("emo"):
            emo_raw = _normalize_emo_text(m.group("emo_text"))
            raw_events.append(("set_emo", emo_raw))
        elif m.group("clear"):
            raw_events.append(("clear_emo", None))
        last = m.end()
    tail = text[last:].strip()
    if tail:
        raw_events.append(("text", tail))
    if not raw_events:
        # no tokens at all
        return [("text", text.strip(), initial_emo)]

    # second pass: apply sticky emo
    current_emo = initial_emo
    for kind, val in raw_events:
        if kind == "text":
            events.append(("text", val, current_emo))
        elif kind == "wait":
            events.append(("wait", val))
        elif kind == "set_emo":
            current_emo = val if val else None
        elif kind == "clear_emo":
            current_emo = None
    return events

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
    steps = []
    remaining = factor
    while remaining < 0.5 or remaining > 2.0:
        if remaining < 1.0:
            steps.append(0.5)
            remaining /= 0.5
        else:
            steps.append(2.0)
            remaining /= 2.0
    steps.append(remaining)
    return ",".join(f"atempo={s:.6f}" for s in steps)

def _stretch_wav(in_path, out_path, factor):
    ffmpeg_bin = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found in PATH.")
    filt = _ffmpeg_atempo_chain(factor)
    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", in_path, "-filter:a", filt, out_path]
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
    parser = argparse.ArgumentParser(description="Batch Text->Speech using IndexTTS (wait + emo tokens + target duration)")
    parser.add_argument("--input", "-i", required=True, help="Text file, one line per utterance. Supports <wait>, <emo: ...>, <emo-clear>, <target ...>.")
    parser.add_argument("--spk_audio_prompt", "-p", required=True, help="Path to speaker prompt audio (wav/mp3)")
    parser.add_argument("--output_path", "-o", required=True, help="Output dir or template. Dir => out/utt_<i>_<name>.wav. Template supports {i} and {name}.")
    parser.add_argument("--emo_alpha", "-a", type=float, default=0.65)
    parser.add_argument("--emo_vector", "-v", default="")
    parser.add_argument("--emo_audio_prompt", help="Path to emotional reference audio (optional)")
    parser.add_argument("--use_emo_text", action="store_true", help="Enable text-driven emotion (used for global/per-line/tokens)")
    parser.add_argument("--use_random", action="store_true")
    parser.add_argument("--emo_text", help="Global default emotion text (used until overridden by <emo ...>)")
    parser.add_argument("--enable_emo_tokens", action="store_true", help="Enable <emo ...> and <emo-clear> sticky tokens")
    parser.add_argument("--line_delim", default="||", help="Backward-compat: 'TEXT || EMO' sets initial emotion for the line (tokens can still override later)")
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

    # parse lines: optional "TEXT || EMO" (for initial emo), also extract inline <target ...>
    lines = []
    inline_targets = []
    delim = args.line_delim
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        initial_emo = None
        text_part = line
        if delim and delim in line:
            # correct order: TEXT || EMO
            text_part, initial_emo = [p.strip() for p in line.split(delim, 1)]
        clean_text, target_sec = _extract_target_and_clean(text_part)
        lines.append((clean_text, initial_emo))
        inline_targets.append(target_sec)

    if not lines:
        print("No non-empty lines found in input.")
        sys.exit(1)

    # target durations
    targets_from_arg = _parse_target_times_arg(args.target_times) if args.target_times else []
    targets_from_file = _read_target_times_file(args.target_times_file) if args.target_times_file else []
    target_seconds = []
    for i in range(len(lines)):
        t = inline_targets[i] if inline_targets[i] is not None else None
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

    provided_globally = bool(
        (args.emo_audio_prompt and str(args.emo_audio_prompt).strip()) or
        (args.emo_vector and str(args.emo_vector).strip()) or
        args.use_emo_text or
        (args.emo_text and str(args.emo_text).strip()) or
        args.use_random
    )

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

    # -------------------- generation with tokens + timing --------------------'
    for idx, (text, initial_emo_for_line) in enumerate(lines, start=1):
        name = _safe_filename(text) or f"line{idx}"
        out_path = make_out(idx, name)

        provided_this_line = provided_globally or (initial_emo_for_line is not None)
        emo_audio_prompt_to_pass = args.emo_audio_prompt if provided_this_line else None
        emo_vec_to_pass = emo_vec if provided_this_line else None
        emo_alpha_to_pass = args.emo_alpha if provided_this_line else 0.0
        use_random_to_pass = args.use_random if provided_this_line else False

        # seed emotion: per-line initial or global
        seed_emo = initial_emo_for_line if initial_emo_for_line else (args.emo_text if args.emo_text else None)
        tgt = target_seconds[idx-1] if idx-1 < len(target_seconds) else None

        try:
            tmpdir = tempfile.mkdtemp(prefix=f"tts_line_{idx:03d}_")
            chunks = []       # [('wav', path) or ('silence', secs)]
            voiced_wavs = []  # wav paths
            waits = []        # float seconds

            # Token path (preferred when enabled) — supports <emo>, <emo-clear>, <wait>
            if (args.enable_emo_tokens or args.enable_wait_tags) and TOKEN_RE.search(text):
                print(f"[{idx}/{len(lines)}] Generating with tokens -> {out_path} (target={tgt}s)")
                seq = _tokenize_with_emo_and_wait(text, initial_emo=seed_emo)
                seg_counter = 0
                for item in seq:
                    if item[0] == "text" and item[1].strip():
                        seg_counter += 1
                        seg_text = item[1]
                        seg_emo  = item[2]

                        # decide emotion flags for this segment
                        seg_use_emo_text = args.use_emo_text and (seg_emo is not None or seed_emo is not None)
                        seg_emo_text_to_pass = seg_emo if seg_emo is not None else (seed_emo if seg_use_emo_text else None)

                        tmp_wav = _infer_one_text_to_temp(
                            tts, seg_text, tmpdir, seg_counter,
                            spk_audio_prompt=args.spk_audio_prompt,
                            emo_audio_prompt=emo_audio_prompt_to_pass,
                            emo_alpha=emo_alpha_to_pass,
                            emo_vector=emo_vec_to_pass,
                            use_emo_text=seg_use_emo_text,
                            emo_text=seg_emo_text_to_pass,
                            use_random=use_random_to_pass,
                            verbose=args.verbose
                        )
                        chunks.append(("wav", tmp_wav))
                        voiced_wavs.append(tmp_wav)

                    elif item[0] == "wait":
                        waits.append(float(item[1]))
                        chunks.append(("silence", float(item[1])))

            else:
                # No tokens path: single segment (still honors global/per-line initial emo)
                print(f"[{idx}/{len(lines)}] Generating -> {out_path} (target={tgt}s)")
                seg_use_emo_text = args.use_emo_text and (seed_emo is not None)
                seg_emo_text_to_pass = seed_emo if seg_use_emo_text else None

                tmp_wav = _infer_one_text_to_temp(
                    tts, text, tmpdir, 1,
                    spk_audio_prompt=args.spk_audio_prompt,
                    emo_audio_prompt=emo_audio_prompt_to_pass,
                    emo_alpha=emo_alpha_to_pass,
                    emo_vector=emo_vec_to_pass,
                    use_emo_text=seg_use_emo_text,
                    emo_text=seg_emo_text_to_pass,
                    use_random=use_random_to_pass,
                    verbose=args.verbose
                )
                chunks.append(("wav", tmp_wav))
                voiced_wavs.append(tmp_wav)

            # --- target duration fitting (same policy as before) ---
            if tgt is not None:
                voiced_dur = sum(_wav_duration_seconds(p) for p in voiced_wavs)
                waits_dur = sum(waits) if waits else 0.0
                total = voiced_dur + waits_dur

                def rebuild_and_export(stretched_voiced_paths, new_waits, path_out):
                    rebuilt, v_idx, w_idx = [], 0, 0
                    for kind, val in chunks:
                        if kind == "wav":
                            rebuilt.append(("wav", stretched_voiced_paths[v_idx]))
                            v_idx += 1
                        else:
                            rebuilt.append(("silence", float(new_waits[w_idx])))
                            w_idx += 1
                    _concat_wavs_with_silences(rebuilt, path_out)
                print(f"  durations: voiced={voiced_dur:.3f}s + waits={waits_dur:.3f}s = total={total:.3f}s")
                print(f"  target: {tgt:.3f}s")
                if total > tgt:
                    need_reduce = total - tgt
                    new_waits = list(waits)
                    if waits_dur > 0:
                        reduce_from_waits = min(need_reduce, waits_dur)
                        scale = (waits_dur - reduce_from_waits) / waits_dur
                        new_waits = [w * scale for w in waits]
                        waits_after = sum(new_waits)
                        remain_delta = tgt - (voiced_dur + waits_after)
                    else:
                        remain_delta = need_reduce
                        new_waits = []
                    print(f"  after waits adjust: new_waits={new_waits}, remain_delta={remain_delta:.3f}s")
                    if remain_delta > 1e-3 and voiced_dur > 0:
                        desired_voiced = max(voiced_dur - remain_delta, 0.05)
                        speed_factor = voiced_dur / desired_voiced  # > 1.0
                        stretched = []
                        for i, vp in enumerate(voiced_wavs, 1):
                            outp = os.path.join(tmpdir, f"seg_speed_{i:04d}.wav")
                            _stretch_wav(vp, outp, speed_factor)
                            stretched.append(outp)
                        rebuild_and_export(stretched, new_waits, out_path)
                    else:
                        rebuild_and_export(voiced_wavs, new_waits, out_path)

                elif total < tgt - 1e-3:
                    need_increase = tgt - total
                    if waits_dur > 0:
                        new_waits = [w + (w / waits_dur) * need_increase for w in waits]
                        rebuild_and_export(voiced_wavs, new_waits, out_path)
                    else:
                        desired_voiced = voiced_dur + need_increase
                        slow_factor = voiced_dur / desired_voiced  # < 1.0
                        slowed = []
                        for i, vp in enumerate(voiced_wavs, 1):
                            outp = os.path.join(tmpdir, f"seg_slow_{i:04d}.wav")
                            _stretch_wav(vp, outp, slow_factor)
                            slowed.append(outp)
                        rebuild_and_export(slowed, [], out_path)
                else:
                    _concat_wavs_with_silences(chunks, out_path)
            else:
                _concat_wavs_with_silences(chunks, out_path)

        except Exception as e:
            print(f"Failed to generate line {idx}: {e}")
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
