"""An optimistic, explicitly partial dense-Qwen3 matrix-compute floor."""


def estimate(case, model, hardware, observed_gpu, measured_ms):
    missing = {
        "status": "unavailable",
        "reason": "Configure a sourced dense BF16 ceiling for the exact GPU in benchmark.yml",
    }
    if not hardware:
        return missing
    if hardware["gpu_name"] != observed_gpu:
        return {
            "status": "unavailable",
            "reason": "Configured GPU ceiling does not match the measured GPU",
        }
    if (
        model.get("model_type") != "qwen3"
        or case["model"]["dtype"] != "bfloat16"
        or case["model"]["quantization"] != "none"
    ):
        return {
            "status": "unavailable",
            "reason": "Estimator currently covers dense Qwen3 BF16 only",
        }
    batch, context, width = (
        case["batch_size"],
        case["context_length"],
        case["beam_width"],
    )
    steps = case["generation"]["output_tokens"] - 1
    layers, hidden, intermediate = (
        model["num_hidden_layers"],
        model["hidden_size"],
        model["intermediate_size"],
    )
    heads, kv_heads = model["num_attention_heads"], model["num_key_value_heads"]
    dim = model.get("head_dim", hidden // heads)
    q, kv = heads * dim, kv_heads * dim
    # FMA = 2 FLOPs. QKV + output projection + gate/up/down projections.
    linear_tokens = batch * (context + steps * width)
    projections = (
        2 * layers * linear_tokens * hidden * (2 * q + 2 * kv + 3 * intermediate)
    )
    lm_head = 2 * batch * (1 + steps * width) * hidden * model["vocab_size"]
    prefill_attention = 2 * batch * layers * heads * dim * context * (context + 1)
    decode_attention = (
        4
        * batch
        * width
        * layers
        * heads
        * dim
        * sum(context + step for step in range(1, steps + 1))
    )
    flops = projections + lm_head + prefill_attention + decode_attention
    floor_ms = flops / (hardware["dense_bf16_tflops"] * 1e12) * 1000
    if measured_ms < floor_ms:
        return {
            "status": "unavailable",
            "reason": "Measurement is below the modeled floor; check assumptions, precision and ceiling",
        }
    return {
        "status": "partial",
        "floor_ms": floor_ms,
        "matrix_flops": flops,
        "bound_over_measured_pct": floor_ms / measured_ms * 100,
        "gap_factor": measured_ms / floor_ms,
        "ceiling_source": hardware["source"],
        "estimator": "dense-qwen3-matrix-v1",
        "assumptions": "Fixed-width full-vocabulary beam search; last-token prefill LM head; ideal dense BF16 rate; causal attention.",
        "limitations": "Matrix math only. Omits memory traffic, softmax, normalization, selection, launch and host costs. The gap is optimistic, not an attainable speedup or GPU utilization.",
    }
