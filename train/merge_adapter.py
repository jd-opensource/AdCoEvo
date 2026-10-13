"""把 LoRA adapter 合并进 base，输出可直接 vLLM serve 的完整模型。
用法：<vllm-venv>/bin/python train/merge_adapter.py <adapter_dir> <out_dir> [base]"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter = sys.argv[1]
out = sys.argv[2]
from train.paths import base_model as _resolve_base
base = str(_resolve_base(sys.argv[3] if len(sys.argv) > 3 else None))
print(f"[merge] base={base} adapter={adapter} -> {out}", flush=True)
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
m = PeftModel.from_pretrained(m, adapter)
m = m.merge_and_unload()
m.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(out)
print("[merge] done", flush=True)
