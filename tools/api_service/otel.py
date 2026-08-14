# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry instrumentation for AIConfigurator API."""

import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
from opentelemetry.sdk.resource import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def init_otel_tracing():
    """Initialize OpenTelemetry tracing with OTLP JSON exporter."""
    otel_endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318')

    resource = Resource.create({
        'service.name': 'aiconfigurator',
        'service.version': '1.0.0',
    })

    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    otlp_exporter = OTLPSpanExporter(endpoint=otel_endpoint)
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    FastAPIInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    URLLib3Instrumentor().instrument()

    logger.info(f'OpenTelemetry initialized with OTLP endpoint: {otel_endpoint}')
