"""
CLIP 텍스트 임베딩 평균 사전 계산 스크립트.
학습 시작마다 재계산하지 않도록 한 번만 실행해두면 됨.
저장 위치: config/clip_mean.pt (실험 이름에 무관하게 공용 사용)
"""
import json
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

JSONL_PATH  = 'config/train.jsonl'
SAVE_PATH   = 'config/clip_mean.pt'
CLIP_MODEL  = 'openai/clip-vit-base-patch32'
MAX_SAMPLES = 10000
BATCH_SIZE  = 256
DEVICE      = 'cuda:0'

tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)
model     = CLIPTextModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()

texts = []
with open(JSONL_PATH) as f:
    for line in f:
        d = json.loads(line)
        for cap in d.get('captions', []):
            for seg in cap.get('segments', []):
                texts.append(seg)
        if len(texts) >= MAX_SAMPLES:
            break
texts = texts[:MAX_SAMPLES]
print(f"총 {len(texts)}개 segment 텍스트로 CLIP mean 계산 중...")

all_embeds = []
with torch.no_grad():
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        tok   = tokenizer(batch, padding=True, truncation=True,
                          max_length=77, return_tensors='pt').to(DEVICE)
        all_embeds.append(model(**tok).pooler_output.cpu())
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  {i}/{len(texts)}")

mean_embed = torch.cat(all_embeds, dim=0).mean(0)  # [512]
torch.save(mean_embed, SAVE_PATH)
print(f"저장 완료: {SAVE_PATH}")

# 검증
all_e = torch.cat(all_embeds, dim=0)
e_norm = F.normalize(all_e, dim=-1)
e_c    = F.normalize(all_e - mean_embed.unsqueeze(0), dim=-1)
n      = min(500, len(e_norm))
mask   = ~torch.eye(n, dtype=torch.bool)
before = (e_norm[:n] @ e_norm[:n].T)[mask]
after  = (e_c[:n]   @ e_c[:n].T)[mask]
print(f"Before centering: mean={before.mean():.3f}  std={before.std():.3f}")
print(f"After  centering: mean={after.mean():.3f}  std={after.std():.3f}")
