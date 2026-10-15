"""OpenTelemetry tracing (Phase 3).

Every agentic step gets a span: which model was chosen, why, tokens, tool
executions. Enabled by setting OTEL_EXPORTER_OTLP_ENDPOINT (OTLP/HTTP JSON
protobuf to any collector — Langfuse, Jaeger, Grafana Tempo). Unset = fully
disabled; zero runtime cost, no network calls.

Env:
  OTEL_EXPORTER_OTLP_ENDPOINT  e.g. http://langfuse:3000/api/public/otel
  OTEL_SERVICE_NAME            default "bridgerouteai"
"""
import logging
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger(__name__)

_ENABLED = False
_tracer = None


def _unwrap_env(v):
    return (v or "").strip()


def init_tracing() -> None:
    """Init OTLP exporter if configured. Never raises."""
    global _ENABLED, _tracer
    endpoint = _unwrap_env(getattr(settings, "otel_exporter_otlp_endpoint", ""))
    if not endpoint:
        logger.info("OTel tracing disabled (OTEL_EXPORTER_OTLP_ENDPOINT unset)")
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        service = _unwrap_env(getattr(settings, "otel_service_name", "")) or "bridgerouteai"
        provider = TracerProvider(resource=Resource.create({"service.name": service}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("bridgerouteai.gateway")
        _ENABLED = True
        logger.info("OTel tracing enabled -> %s (service=%s)", endpoint, service)
    except ImportError as exc:
        logger.warning("OTel packages not installed — tracing disabled: %s", exc)
    except Exception as exc:
        logger.warning("OTel init failed — tracing disabled: %s", exc)


@contextmanager
def span(name: str, attributes: dict | None = None):
    """Context-managed span. No-op when tracing is disabled or init failed."""
    if not _ENABLED or _tracer is None:
        yield None
        return
    s = _tracer.start_span(name, attributes=attributes or {})
    try:
        yield s
        s.end()
    except Exception as exc:
        s.record_exception(exc)
        s.end()
        raise


def set_attrs(s, attributes: dict) -> None:
    if s is None:
        return
    try:
        for k, v in (attributes or {}).items():
            if v is not None:
                s.set_attribute(k, str(v)[:400] if isinstance(v, str) else v)
    except Exception:
        pass
