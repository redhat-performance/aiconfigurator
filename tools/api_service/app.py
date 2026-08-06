# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIConfigurator REST API.

Minimal service wrapping the aiconfigurator SDK for GPU recommendation
and memory estimation.  See docs/api/openapi.yaml for the full spec.
"""

import argparse
import logging
import sys
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, model_validator

from aiconfigurator.cli.api import cli_recommend
from aiconfigurator.sdk.common import get_default_models
from aiconfigurator.sdk.memory import estimate_kv_cache
from aiconfigurator_core.sdk.common import SupportedSystems
from aiconfigurator_core.sdk.perf_database import load_system_spec

logger = logging.getLogger(__name__)

# ─── Pydantic models ────────────────────────────────────────────────────────


class RecommendRequest(BaseModel):
    model_path: str = Field(examples=["Qwen/Qwen3-32B"], description="HuggingFace model path or SDK model key.")
    system: str = Field(examples=["h200_sxm"], description="GPU system identifier.")
    backend: str = Field(default="vllm", description="Inference backend.")
    backend_version: str | None = Field(default=None, examples=[None], description="Backend version.")
    target_request_rate: float | None = Field(default=None, examples=[None], description="Target req/s.")
    target_concurrency: float | None = Field(default=None, examples=[32], description="Target concurrent users.")
    isl: int = Field(default=4000, description="Input sequence length.")
    osl: int = Field(default=1000, description="Output sequence length.")
    ttft: float = Field(default=2000.0, description="TTFT target (ms).")
    tpot: float = Field(default=30.0, description="TPOT target (ms).")
    request_latency: float | None = Field(default=None, description="E2E latency target (ms).")
    prefix: int = Field(default=0, description="Prefix cache length.")
    database_mode: str = Field(default="HYBRID", description="Perf database mode.")
    top_n: int = Field(default=5, ge=1, le=20, examples=[2], description="Number of configs to return.")

    @model_validator(mode="after")
    def exactly_one_load_target(self):
        has_rate = self.target_request_rate is not None
        has_conc = self.target_concurrency is not None
        if has_rate == has_conc:
            raise ValueError("Exactly one of target_request_rate or target_concurrency must be provided.")
        return self


class MemoryBreakdown(BaseModel):
    weights_bytes: int = 0
    activations_bytes: int = 0
    runtime_overhead_bytes: int = 0
    comm_overhead_bytes: int = 0
    kv_cache_bytes: int = 0


class ServingConfig(BaseModel):
    backend: str
    tensor_parallel_size: int
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    enable_chunked_prefill: bool
    enable_prefix_caching: bool
    quantization: str


class RecommendConfig(BaseModel):
    total_gpus_needed: int | None = None
    replicas_needed: int | None = None
    num_total_gpus: int | None = None
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    moe_tp: int | None = None
    moe_ep: int | None = None
    cp: int | None = None
    ttft: float | None = None
    tpot: float | None = None
    request_latency: float | None = None
    concurrency: int | None = None
    request_rate: float | None = None
    tokens_per_second: float | None = None
    tokens_per_second_per_gpu: float | None = None
    tokens_per_second_per_user: float | None = None
    memory: float | None = None
    model: str | None = None
    system: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    gemm: str | None = None
    kvcache: str | None = None
    fmha: str | None = None
    moe: str | None = None
    comm: str | None = None
    power_w: float | None = None
    serving_config: ServingConfig | None = None
    memory_breakdown: MemoryBreakdown | None = None


class RecommendResponse(BaseModel):
    configs: list[RecommendConfig]
    chosen_mode: str


class MemoryRequest(BaseModel):
    model_path: str = Field(examples=["Qwen/Qwen3-32B"], description="HuggingFace model path or SDK model key.")
    system: str = Field(examples=["h200_sxm"], description="GPU system identifier.")
    backend: str = Field(default="vllm", description="Inference backend.")
    backend_version: str | None = Field(default=None, examples=[None])
    max_num_tokens: int = Field(default=8192)
    max_batch_size: int = Field(default=128)
    memory_fraction_kind: str = Field(default="of_total")
    memory_fraction_value: float = Field(default=1.0, ge=0.0, le=1.0)
    tp_size: int = Field(default=1)
    pp_size: int = Field(default=1)
    attention_dp_size: int = Field(default=1)
    moe_tp_size: int | None = Field(default=None)
    moe_ep_size: int | None = Field(default=None)
    gemm_quant_mode: str | None = Field(default=None)
    moe_quant_mode: str | None = Field(default=None)
    kvcache_quant_mode: str | None = Field(default=None)
    fmha_quant_mode: str | None = Field(default=None)
    comm_quant_mode: str | None = Field(default=None)
    tolerance_fraction: float | None = Field(default=None)


class MemoryResponse(BaseModel):
    total_gpu_capacity_bytes: int
    total_kv_size_bytes: int
    kv_size_per_token_bytes: int
    total_kv_size_tokens: int
    source: str
    memory_breakdown: MemoryBreakdown
    tolerance_adjusted: dict[str, Any] | None = None


class SystemDetail(BaseModel):
    id: str
    name: str
    vendor: str
    architecture: str
    memory_bytes: int
    tdp_watts: float
    gpus_per_node: int


# ─── Helpers ─────────────────────────────────────────────────────────────────

_COLUMN_MAP = {
    "tokens/s": "tokens_per_second",
    "tokens/s/gpu": "tokens_per_second_per_gpu",
    "tokens/s/user": "tokens_per_second_per_user",
    "num_total_gpus": "num_total_gpus",
    "version": "backend_version",
}

_INT_FIELDS = frozenset({
    "total_gpus_needed", "replicas_needed", "num_total_gpus",
    "tp", "pp", "dp", "moe_tp", "moe_ep", "cp", "concurrency",
})

_SM_ARCHITECTURE = {
    70: "volta",
    75: "turing",
    80: "ampere",
    86: "ampere",
    89: "ada-lovelace",
    90: "hopper",
    100: "blackwell",
    120: "blackwell",
}

_DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


def _row_to_config(row: pd.Series, req: RecommendRequest) -> RecommendConfig:
    d: dict[str, Any] = {}
    for col, val in row.items():
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        key = _COLUMN_MAP.get(str(col), str(col))
        if isinstance(val, float) and val == int(val) and key in _INT_FIELDS:
            val = int(val)
        d[key] = val
    d.setdefault("system", req.system)
    d.setdefault("backend", req.backend)
    d.setdefault("backend_version", req.backend_version)
    return RecommendConfig.model_validate(d)


def _build_serving_config(
    config: RecommendConfig,
    req: RecommendRequest,
) -> ServingConfig:
    """Build serving config from a recommend result."""
    tp = config.tp or 1
    backend = config.backend or req.backend
    gemm_quant = config.gemm or "bfloat16"
    max_model_len = req.isl + req.osl
    concurrency = config.concurrency or 1

    quant_map = {
        "fp8": "fp8",
        "fp8_block": "fp8",
        "int8": "int8",
        "bfloat16": None,
        "half": None,
    }
    quantization = quant_map.get(gemm_quant)

    use_chunked_prefill = req.isl >= 4096 or concurrency >= 64

    return ServingConfig(
        backend=backend,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        max_num_seqs=min(concurrency, 256),
        gpu_memory_utilization=_DEFAULT_GPU_MEMORY_UTILIZATION,
        enable_chunked_prefill=use_chunked_prefill,
        enable_prefix_caching=req.prefix > 0,
        quantization=quantization or "auto",
    )


def _build_memory_breakdown(
    req: RecommendRequest,
    config: RecommendConfig,
) -> MemoryBreakdown | None:
    """Run estimate_kv_cache for a recommend result and return breakdown."""
    tp = config.tp or 1
    pp = config.pp or 1
    backend_version = config.backend_version or req.backend_version
    gemm_quant = config.gemm if config.gemm and config.gemm != "half" else None
    kvcache_quant = config.kvcache if config.kvcache and config.kvcache != "half" else None

    try:
        raw = estimate_kv_cache(
            model_path=req.model_path,
            system=req.system,
            backend=config.backend or req.backend,
            backend_version=backend_version,
            max_num_tokens=req.isl + req.osl,
            max_batch_size=config.concurrency or 128,
            memory_fraction_kind="of_total",
            memory_fraction_value=_DEFAULT_GPU_MEMORY_UTILIZATION,
            tp_size=tp,
            pp_size=pp,
            moe_tp_size=config.moe_tp,
            moe_ep_size=config.moe_ep,
            gemm_quant_mode=gemm_quant,
            kvcache_quant_mode=kvcache_quant,
        )
    except Exception:
        logger.debug("memory breakdown unavailable for %s on %s", req.model_path, req.system)
        return None

    breakdown = raw.get("memory_breakdown") or {}
    return MemoryBreakdown(
        weights_bytes=int(breakdown.get("weights_bytes", 0)),
        activations_bytes=int(breakdown.get("activations_bytes", 0)),
        runtime_overhead_bytes=int(breakdown.get("runtime_overhead_bytes", 0)),
        comm_overhead_bytes=int(breakdown.get("comm_overhead_bytes", 0)),
        kv_cache_bytes=int(raw.get("total_kv_size_bytes", 0)),
    )


def _architecture_from_sm(sm_version: int) -> str:
    return _SM_ARCHITECTURE.get(sm_version, f"sm_{sm_version}")


def _system_display_name(system_id: str) -> str:
    parts = system_id.replace("_", " ").upper().split()
    return "NVIDIA " + " ".join(parts)


def _parse_include(include: str | None) -> set[str]:
    if not include:
        return set()
    return {s.strip().lower() for s in include.split(",")}


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AIConfigurator API",
    description="GPU recommendation and memory estimation for LLM inference.",
    version="1.0.0",
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/recommend")
def post_recommend(
    req: RecommendRequest,
    include: str | None = Query(default=None, examples=["config,memory"], description="Comma-separated extras: config, memory."),
):
    """Find optimal GPU configuration for a workload."""
    try:
        result = cli_recommend(
            model_path=req.model_path,
            system=req.system,
            backend=req.backend,
            backend_version=req.backend_version,
            target_request_rate=req.target_request_rate,
            target_concurrency=req.target_concurrency,
            database_mode=req.database_mode,
            isl=req.isl,
            osl=req.osl,
            ttft=req.ttft,
            tpot=req.tpot,
            request_latency=req.request_latency,
            prefix=req.prefix,
            top_n=req.top_n,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except AttributeError as e:
        msg = str(e)
        if "system_spec" in msg or "NoneType" in msg:
            raise HTTPException(
                status_code=422,
                detail=f"No performance data available for model={req.model_path}, backend={req.backend}, system={req.system}.",
            )
        logger.exception("recommend failed")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("recommend failed")
        raise HTTPException(status_code=500, detail=str(e))

    best = result.best_configs.get(result.chosen_exp)
    if best is None or best.empty:
        raise HTTPException(status_code=422, detail="No configuration meets the specified requirements.")

    includes = _parse_include(include)
    want_config = "config" in includes
    want_memory = "memory" in includes

    configs = []
    for _, row in best.head(req.top_n).iterrows():
        cfg = _row_to_config(row, req)
        if want_config:
            cfg.serving_config = _build_serving_config(cfg, req)
        if want_memory:
            cfg.memory_breakdown = _build_memory_breakdown(req, cfg)
        configs.append(cfg)

    return RecommendResponse(configs=configs, chosen_mode=result.chosen_exp)


@app.post("/memory", response_model=MemoryResponse)
def post_memory(req: MemoryRequest):
    """Estimate GPU memory breakdown for a model configuration."""
    try:
        raw = estimate_kv_cache(
            model_path=req.model_path,
            system=req.system,
            backend=req.backend,
            backend_version=req.backend_version,
            max_num_tokens=req.max_num_tokens,
            max_batch_size=req.max_batch_size,
            memory_fraction_kind=req.memory_fraction_kind,
            memory_fraction_value=req.memory_fraction_value,
            tp_size=req.tp_size,
            pp_size=req.pp_size,
            attention_dp_size=req.attention_dp_size,
            moe_tp_size=req.moe_tp_size,
            moe_ep_size=req.moe_ep_size,
            gemm_quant_mode=req.gemm_quant_mode,
            moe_quant_mode=req.moe_quant_mode,
            kvcache_quant_mode=req.kvcache_quant_mode,
            fmha_quant_mode=req.fmha_quant_mode,
            comm_quant_mode=req.comm_quant_mode,
            tolerance_fraction=req.tolerance_fraction,
        )
    except (ValueError, AttributeError) as e:
        msg = str(e)
        if "system_spec" in msg or "NoneType" in msg or "unsupported model" in msg.lower():
            detail = (
                f"No performance data available for model={req.model_path}, "
                f"backend={req.backend}, system={req.system}"
            )
            if not req.backend_version:
                detail += ". Try specifying a backend_version (e.g. '0.24.0' for vllm)."
            raise HTTPException(status_code=422, detail=detail)
        raise HTTPException(status_code=422, detail=msg)
    except Exception as e:
        logger.exception("memory estimation failed")
        raise HTTPException(status_code=500, detail=str(e))

    breakdown = raw.get("memory_breakdown") or {}
    return MemoryResponse(
        total_gpu_capacity_bytes=raw["total_gpu_capacity_bytes"],
        total_kv_size_bytes=raw["total_kv_size_bytes"],
        kv_size_per_token_bytes=raw["kv_size_per_token_bytes"],
        total_kv_size_tokens=raw["total_kv_size_tokens"],
        source=raw.get("source", "unknown"),
        memory_breakdown=MemoryBreakdown(
            weights_bytes=int(breakdown.get("weights_bytes", 0)),
            activations_bytes=int(breakdown.get("activations_bytes", 0)),
            runtime_overhead_bytes=int(breakdown.get("runtime_overhead_bytes", 0)),
            comm_overhead_bytes=int(breakdown.get("comm_overhead_bytes", 0)),
            kv_cache_bytes=raw["total_kv_size_bytes"],
        ),
        tolerance_adjusted=raw.get("tolerance_adjusted"),
    )


@app.get("/models")
def get_models():
    """List supported models."""
    return {"models": sorted(get_default_models())}


@app.get("/systems")
def get_systems(
    include: str | None = Query(default=None, examples=["specs"], description="Comma-separated extras: specs."),
):
    """List supported GPU systems."""
    includes = _parse_include(include)
    want_specs = "specs" in includes

    systems = []
    for sys_id in sorted(SupportedSystems):
        entry: dict[str, Any] = {
            "id": sys_id,
            "name": _system_display_name(sys_id),
        }
        if want_specs:
            try:
                spec = load_system_spec(sys_id)
                gpu = spec.get("gpu", {})
                node = spec.get("node", {})
                sm = int(gpu.get("sm_version", 0))
                entry.update({
                    "vendor": "nvidia",
                    "architecture": _architecture_from_sm(sm),
                    "memory_bytes": int(gpu.get("mem_capacity", 0)),
                    "tdp_watts": float(gpu.get("power", 0)),
                    "gpus_per_node": int(node.get("num_gpus_per_node", 0)),
                })
            except Exception:
                logger.warning("failed to load spec for %s", sys_id)
        systems.append(entry)
    return {"systems": systems}


# ─── Entrypoint ──────────────────────────────────────────────────────────────

def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    return parser.parse_args(args=args)


if __name__ == "__main__":
    args = parse(sys.argv[1:])
    uvicorn.run(app, host=args.server_name, port=args.server_port)
