from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = "/home/model/LLM/LGU-ixi-GEN/260526_ixi_GEN_1.2B_vocab_trim"
TEST_DATA_DIR = "/home/kh_project/02_quant/test_data"

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# 테스트 데이터 추론
test_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "*.txt")))

for fpath in test_files:
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(fpath)}")
    print('='*60)

    with open(fpath, 'r') as f:
        prompt = f.read().strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            eos_token_id=tokenizer.convert_tokens_to_ids("[|endofturn|]") if "[|endofturn|]" in tokenizer.get_vocab() else tokenizer.eos_token_id
        )

    generated = tokenizer.decode(
        outputs[0][len(inputs['input_ids'][0]):],
        skip_special_tokens=False
    )
    result = generated.split('[|endofturn|]', 1)[0].strip()

    print(f"\n--- Generated Output ---")
    print(result)
    print(f"--- End ---\n")
