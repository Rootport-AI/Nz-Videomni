"""Stand-in for diffsynth.core.gradient.gradient_checkpoint_forward (pass-through)."""


def gradient_checkpoint_forward(model, use_gradient_checkpointing, use_gradient_checkpointing_offload, *args, **kwargs):
    return model(*args, **kwargs)
