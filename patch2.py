import os
import math

with open('c:/Users/ravin/Downloads/OPENAI/parameter-golf/records/track_10min_16mb/our_submission/train_gpt.py', 'r') as f:
    code = f.read()

# 1. Hyperparams - Add rope_dims
old_hp = '''    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))'''

new_hp = '''    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))'''
code = code.replace(old_hp, new_hp)

# Update GPT instantiation
code = code.replace('rope_base = rope_base,', 'rope_base,')
code = code.replace('qk_gain_init = qk_gain_init,', 'qk_gain_init,')

# 2. Add ROPE_DIMS dynamically to block
old_block_attn = '''        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)'''
new_block_attn = '''        rope_dims = int(os.environ.get("ROPE_DIMS", 16))
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims)'''
code = code.replace(old_block_attn, new_block_attn)

# update __init__ of CausalSelfAttention
old_csa_def = '''    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.rope_dims = 16'''
new_csa_def = '''    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 16,
    ):
        super().__init__()
        self.rope_dims = rope_dims'''
code = code.replace(old_csa_def, new_csa_def)

code = code.replace('''                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )''', '''                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )''')

# 3. Tied Embedding QAT
old_tied_qat = '''        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)'''
new_tied_qat = '''        if self.tie_embeddings:
            w = self.tok_emb.weight
            if self.training:
                # Per-channel Fake QAT for tied embeddings
                scale = (w.abs().amax(dim=1, keepdim=True) / 31.0).clamp_min(1e-12)
                w = (w / scale).round().clamp(-31.0, 31.0) * scale
            logits_proj = F.linear(x, w)'''
code = code.replace(old_tied_qat, new_tied_qat)

# 4. Correct Sliding Window eval_val logic
old_eval_val_v2 = '''    model.eval()
    with torch.inference_mode():
        stride = 64
        for batch_seq_start in range(seq_start, seq_end - 1, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end - 1)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss_unreduced = model(x, y, reduction="none").detach()
            
            loss_stride = loss_unreduced.view(-1, args.train_seq_len)[:, -stride:]
            batch_loss = loss_stride.sum()
            batch_token_count = float(loss_stride.numel())
            
            val_loss_sum += batch_loss.to(torch.float64)
            val_token_count += batch_token_count
            
            prev_ids = x[:, -stride:].reshape(-1)
            tgt_ids = y[:, -stride:].reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()'''

new_eval_val_v2 = '''    model.eval()
    with torch.inference_mode():
        stride = 64
        seq_len = args.train_seq_len
        raw_start = seq_start * seq_len
        raw_end = seq_end * seq_len
        
        local_tokens = val_tokens[raw_start : raw_end + 1].to(device=device, dtype=torch.int64, non_blocking=True)
        # Handle cases where rank slice is smaller than seq_len
        if local_tokens.size(0) > seq_len:
            windows = local_tokens.unfold(0, seq_len + 1, stride)
            for i in range(0, windows.size(0), local_batch_seqs):
                batch_windows = windows[i : i + local_batch_seqs]
                x = batch_windows[:, :-1]
                y = batch_windows[:, 1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss_unreduced = model(x, y, reduction="none").detach()
                
                loss_stride = loss_unreduced[:, -stride:]
                batch_loss = loss_stride.sum()
                batch_token_count = float(loss_stride.numel())
                
                val_loss_sum += batch_loss.to(torch.float64)
                val_token_count += batch_token_count
                
                prev_ids = x[:, -stride:].reshape(-1)
                tgt_ids = y[:, -stride:].reshape(-1)
                token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                val_byte_count += token_bytes.to(torch.float64).sum()'''
code = code.replace(old_eval_val_v2, new_eval_val_v2)

# 5. EMA swap fix
old_swap_v2 = '''            # EMA Swap for validation
            train_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
            base_model.load_state_dict(ema_state, strict=True)'''
new_swap_v2 = '''            # EMA Swap for validation via deepcopy (keeps optim stats intact)
            import copy
            backup_state = copy.deepcopy(base_model.state_dict())
            base_model.load_state_dict(ema_state, strict=True)'''
code = code.replace(old_swap_v2, new_swap_v2)

old_swap3 = '''            base_model.load_state_dict(train_state, strict=True)'''
new_swap3 = '''            base_model.load_state_dict(backup_state, strict=True)'''
code = code.replace(old_swap3, new_swap3)

with open('c:/Users/ravin/Downloads/OPENAI/parameter-golf/records/track_10min_16mb/our_submission/train_gpt.py', 'w') as f:
    f.write(code)

print('Patch 2 applied successfully')
