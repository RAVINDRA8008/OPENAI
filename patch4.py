import os

with open('c:/Users/ravin/Downloads/OPENAI/parameter-golf/records/track_10min_16mb/our_submission/train_gpt.py', 'r') as f:
    code = f.read()

old_swap = '''            # EMA Swap for validation via deepcopy (keeps optim stats intact)
            import copy
            backup_state = copy.deepcopy(base_model.state_dict())
            base_model.load_state_dict(ema_state, strict=True)'''

new_swap = '''            # Instant memory-free EMA pointer swap
            train_state_cache = {}
            for k, v in base_model.state_dict().items():
                train_state_cache[k] = v.data
                v.data = ema_state[k]'''
                
# Wait, ema_state is a dict of tensors. So ema_state[k] is a tensor. 
# v.data = ema_state[k].data is safer. But wait, ema_state[k] is just a tensor.
code = code.replace(old_swap, new_swap)

old_restore = '''            base_model.load_state_dict(backup_state, strict=True)'''

new_restore = '''            for k, v in base_model.state_dict().items():
                v.data = train_state_cache[k]'''

code = code.replace(old_restore, new_restore)

with open('c:/Users/ravin/Downloads/OPENAI/parameter-golf/records/track_10min_16mb/our_submission/train_gpt.py', 'w') as f:
    f.write(code)

print('Patch 4 (Zero-memory EMA pointer swap) applied successfully.')
