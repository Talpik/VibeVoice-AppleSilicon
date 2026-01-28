#!/usr/bin/env python3
"""CLI helper to run batched generation in a subprocess and return results via JSON.

Usage: python run_batched_subprocess.py <input.json> <output.json>
"""
import json
import sys
import traceback

def main():
    if len(sys.argv) < 3:
        print("Usage: run_batched_subprocess.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(2)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception as e:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"error": f"Failed to read input: {e}"}, f)
        sys.exit(1)

    try:
        # Import heavy modules inside the process (may take time but isolates memory)
        from gradio_demo import VibeVoiceDemo
    except Exception:
        # Try package import fallback
        try:
            from demo.gradio_demo import VibeVoiceDemo
        except Exception:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"error": "Cannot import VibeVoiceDemo from demo.gradio_demo"}, f)
            sys.exit(1)

    try:
        demo = VibeVoiceDemo(
            model_path=payload.get('model_path'),
            device=payload.get('device', 'cuda'),
            inference_steps=payload.get('inference_steps', 10),
            adapter_path=payload.get('adapter_path', None),
        )

        out_path, final_log = demo.generate_podcast_batched(
            num_speakers=payload.get('num_speakers'),
            script=payload.get('script',''),
            speaker_1=payload.get('speaker_1'),
            speaker_2=payload.get('speaker_2'),
            speaker_3=payload.get('speaker_3'),
            speaker_4=payload.get('speaker_4'),
            cfg_scale=payload.get('cfg_scale', 1.3),
            inference_steps=payload.get('inference_steps'),
            seed=payload.get('seed'),
            disable_voice_cloning=payload.get('disable_voice_cloning', False),
            stability_preset=payload.get('stability_preset','Natural')
        )

        result = {
            "out_path": out_path,
            "final_log": final_log,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f)
        sys.exit(0)

    except Exception as e:
        tb = traceback.format_exc()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"error": str(e), "traceback": tb}, f)
        sys.exit(1)

if __name__ == '__main__':
    main()
