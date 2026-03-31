import os

with open('c:/Users/ravin/Downloads/OPENAI/parameter-golf/records/track_10min_16mb/our_submission/train_gpt.py', 'r') as f:
    code = f.read()

# 1. CastedLinear FP32 QAT
old_linear_qat = '''        w = self.weight.to(x.dtype)
        if self.training and w.ndim == 2:
            scale = (w.abs().amax(dim=1, keepdim=True) / 31.0).clamp_min(1e-12)
            w = (w / scale).round().clamp(-31.0, 31.0) * scale
        return F.linear(x, w, bias)'''

new_linear_qat = '''        w_dtype = x.dtype
        w = self.weight.to(w_dtype)
        if self.training and w.ndim == 2:
            w32 = w.float()
            scale = (w32.abs().amax(dim=1, keepdim=True) / 31.0).clamp_min(1e-12)
            w_q = (w32 / scale).round().clamp(-31.0, 31.0) * scale
            w = w_q.to(w_dtype)
        return F.linear(x, w, bias)'''
code = code.replace(old_linear_qat, new_linear_qat)

# 2. Tied QAT FP32 
old_tied_qat = '''        if self.tie_embeddings:
            w = self.tok_emb.weight
            if self.training:
                # Per-channel Fake QAT for tied embeddings
                scale = (w.abs().amax(dim=1, keepdim=True) / 31.0).clamp_min(1e-12)
                w = (w / scale).round().clamp(-31.0, 31.0) * scale
            logits_proj = F.linear(x, w)'''

new_tied_qat = '''        if self.tie_embeddings:
            w = self.tok_emb.weight
            if self.training:
                # Per-channel Fake QAT for tied embeddings in fp32
                w32 = w.float()
                scale = (w32.abs().amax(dim=1, keepdim=True) / 31.0).clamp_min(1e-12)
                w_q = (w32 / scale).round().clamp(-31.0, 31.0) * scale
                w = w_q.to(w.dtype)
            logits_proj = F.linear(x, w)'''
code = code.replace(old_tied_qat, new_tied_qat)

# 3. EMA Delay (Wait 200 steps to start averaging)
old_ema_update = '''        with torch.no_grad():
            for k, v in base_model.state_dict().items():
                ema_state[k].mul_(0.9995).add_(v, alpha=0.0005)

        step += 1'''

new_ema_update = '''        with torch.no_grad():
            if step < 200:
                for k, v in base_model.state_dict().items():
                    ema_state[k].copy_(v)
            else:
                for k, v in base_model.state_dict().items():
                    ema_state[k].mul_(0.9995).add_(v, alpha=0.0005)

        step += 1'''
code = code.replace(old_ema_update, new_ema_update)

with open('c:/Users/ravin/Downloads/OPENAI/parameter-golf/records/track_10min_16mb/our_submission/train_gpt.py', 'w') as f:
    f.write(code)

print('Patch 3 (Final FP32 and EMA constraints) applied successfully.')
