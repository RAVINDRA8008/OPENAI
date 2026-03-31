import os
import math

with open('c:/Users/ravin/Downloads/OPENAI/parameter-golf/records/track_10min_16mb/our_submission/train_gpt.py', 'r') as f:
    code = f.read()

# 1. Hyperparameters
old_hp = '''    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))'''

new_hp = '''    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 15000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 400))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 36000.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))'''
code = code.replace(old_hp, new_hp)

# 2. eval_val sliding window
old_eval = '''    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()'''

new_eval = '''    model.eval()
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
code = code.replace(old_eval, new_eval)

# 3. GPT forward params
old_gpt_fwd = '''    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:'''
new_gpt_fwd = '''    def forward(self, input_ids: Tensor, target_ids: Tensor, reduction: str = "mean") -> Tensor:'''
code = code.replace(old_gpt_fwd, new_gpt_fwd)

old_gpt_loss = '''        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")'''
new_gpt_loss = '''        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction=reduction)'''
code = code.replace(old_gpt_loss, new_gpt_loss)

# 4. WD optimizer
old_opt = '''    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )'''
new_opt = '''    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, 0.99),
        eps=args.adam_eps,
        weight_decay=0.04,
        fused=True,
    )'''
code = code.replace(old_opt, new_opt)

# 5. EMA tracking
old_loop_start = '''    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}'''
new_loop_start = '''    import math
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}'''
code = code.replace(old_loop_start, new_loop_start)

old_main = '''        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP'''
new_main = '''        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    ema_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

    # -----------------------------
    # MAIN TRAINING LOOP'''
code = code.replace(old_main, new_main)

old_step = '''        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1'''
new_step = '''        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        with torch.no_grad():
            for k, v in base_model.state_dict().items():
                ema_state[k].mul_(0.9995).add_(v, alpha=0.0005)

        step += 1'''
code = code.replace(old_step, new_step)

old_swap = '''        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val('''
new_swap = '''        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            
            # EMA Swap for validation
            train_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
            base_model.load_state_dict(ema_state, strict=True)
            
            val_loss, val_bpb = eval_val('''
code = code.replace(old_swap, new_swap)

old_swap2 = '''                is_boundary_token_lut,
            )
            log0('''
new_swap2 = '''                is_boundary_token_lut,
            )
            
            base_model.load_state_dict(train_state, strict=True)
            
            log0('''
code = code.replace(old_swap2, new_swap2)

old_quant = '''    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())'''
new_quant = '''    if master_process:
        torch.save(ema_state, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(ema_state)'''
code = code.replace(old_quant, new_quant)

# 6. lr_mul schedule
old_lr = '''    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0'''
new_lr = '''    def lr_mul(step: int, elapsed_ms: float) -> float:
        if step < args.warmup_steps:
            return float(step) / max(1, args.warmup_steps)
        total = args.iterations
        progress = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_scale = 0.1
        return min_scale + (1.0 - min_scale) * cosine'''
code = code.replace(old_lr, new_lr)

with open('c:/Users/ravin/Downloads/OPENAI/parameter-golf/records/track_10min_16mb/our_submission/train_gpt.py', 'w') as f:
    f.write(code)

print('Patch application complete.')
