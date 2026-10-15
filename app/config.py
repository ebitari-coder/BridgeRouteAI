from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Redis
    redis_url: str = Field(..., alias="REDIS_URL")

    # PostgreSQL
    database_url: str = Field(..., alias="DATABASE_URL")

    # Admin API Key for protected endpoints
    admin_api_key: str = Field(..., alias="ADMIN_API_KEY")

    # JWT Auth — NO default, must be set in env
    jwt_secret: str = Field(..., alias="JWT_SECRET")
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 72

    # CORS — comma-separated origins
    cors_origins: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")

    # Superadmin — NO default, must be set in env
    superadmin_email: str = Field(default="admin@bridgeroute.local", alias="SUPERADMIN_EMAIL")
    superadmin_password: str = Field(..., alias="SUPERADMIN_PASSWORD")

    # Kilo AI Gateway (primary upstream)
    kilo_base_url: str = Field(
        default="https://api.kilo.ai/api/gateway",
        alias="KILO_BASE_URL",
    )
    # Optional — free models work anonymously, a key raises rate limits
    kilo_api_key: str = Field(default="", alias="KILO_API_KEY")
    # How often the background task refreshes the Kilo model catalog (seconds)
    kilo_catalog_refresh_seconds: int = Field(default=300, alias="KILO_CATALOG_REFRESH_SECONDS")

    # FreeModels provider (fallback upstream)
    freemodels_url: str = Field(
        default="https://freemodels-chat.freemodels.workers.dev",
        alias="FREE_MODELS_URL",
    )
    # Official site — its SPA bundle carries the live model catalog we scrape
    freemodels_site_url: str = Field(
        default="https://freemodels.pro",
        alias="FREE_MODELS_SITE_URL",
    )
    # How often the background task re-scrapes the FreeModels catalog (seconds)
    freemodels_catalog_refresh_seconds: int = Field(
        default=300, alias="FREE_MODELS_CATALOG_REFRESH_SECONDS"
    )

    # Composite (Lucidity) — OpenAI-compatible gateway; free models end in `:free`
    composite_api_key: str = Field(default="", alias="COMPOSITE_API_KEY")
    composite_base_url: str = Field(
        default="https://composite.lucidity.sh/v1", alias="COMPOSITE_BASE_URL"
    )
    composite_catalog_refresh_seconds: int = Field(
        default=300, alias="COMPOSITE_CATALOG_REFRESH_SECONDS"
    )

    # Septor Labs — OpenAI-compatible gateway; free models end in `-free`
    septor_api_key: str = Field(default="", alias="SEPTOR_API_KEY")
    septor_base_url: str = Field(
        default="https://api.septorlabs.com/v1", alias="SEPTOR_BASE_URL"
    )
    # Public catalog page (no auth) — free models are the `-free` slugs
    septor_site_url: str = Field(
        default="https://septorlabs.com", alias="SEPTOR_SITE_URL"
    )
    septor_catalog_refresh_seconds: int = Field(
        default=300, alias="SEPTOR_CATALOG_REFRESH_SECONDS"
    )

    # Routeway — OpenAI-compatible gateway; free models end in `:free`
    # (catalog endpoint is public; chat needs ROUTEWAY_API_KEY)
    routeway_api_key: str = Field(default="", alias="ROUTEWAY_API_KEY")
    routeway_base_url: str = Field(
        default="https://api.routeway.ai/v1", alias="ROUTEWAY_BASE_URL"
    )
    routeway_catalog_refresh_seconds: int = Field(
        default=300, alias="ROUTEWAY_CATALOG_REFRESH_SECONDS"
    )

    # Cloudflare Workers AI (free tier, requires account + API key)
    cloudflare_account_id: str = Field(default="", alias="CLOUDFLARE_ACCOUNT_ID")
    cloudflare_api_key: str = Field(default="", alias="CLOUDFLARE_API_KEY")

    # BazaarLink (OpenAI-compatible gateway, free tier)
    bazaarlink_api_key: str = Field(default="", alias="BAZAARLINK_API_KEY")
    bazaarlink_enabled: bool = Field(default=False, alias="BAZAARLINK_ENABLED")

    # KiosAPI (OpenAI-compatible gateway, free tier with tool calling)
    kiosapi_api_key: str = Field(default="", alias="KIOSAPI_API_KEY")

    # Xkiro (OpenAI-compatible gateway, 37+ free models with tool calling)
    xkiro_api_key: str = Field(default="", alias="XKIRO_API_KEY")
    xkiro_stability_check_seconds: int = Field(default=300, alias="XKIRO_STABILITY_CHECK_SECONDS")

    # ModelScope (API-Inference, 34+ free open-source LLMs)
    modelscope_token: str = Field(default="", alias="MODELSCOPE_TOKEN")
    modelscope_stability_check_seconds: int = Field(default=300, alias="MODELSCOPE_STABILITY_CHECK_SECONDS")

    # UnoRouter (OpenAI-compatible gateway, 132 free models)
    unorouter_api_key: str = Field(default="", alias="UNOROUTER_API_KEY")
    unorouter_stability_check_seconds: int = Field(default=300, alias="UNOROUTER_STABILITY_CHECK_SECONDS")

    # LLM7 (AI gateway, 6 free turbo models)
    llm7_api_key: str = Field(default="", alias="LLM7_API_KEY")
    llm7_stability_check_seconds: int = Field(default=300, alias="LLM7_STABILITY_CHECK_SECONDS")

    # Agnes AI (2 free models, $0/M tokens, 512K context)
    agnes_api_key: str = Field(default="", alias="AGNES_API_KEY")

    # Top Tools AI (OpenAI-compatible, single model with vision, 512K context)
    toptools_api_key: str = Field(default="", alias="TOPTOOLS_API_KEY")
    toptools_base_url: str = Field(
        default="https://top-tools-ai.com/v1", alias="TOPTOOLS_BASE_URL"
    )
    toptools_catalog_refresh_seconds: int = Field(
        default=300, alias="TOPTOOLS_CATALOG_REFRESH_SECONDS"
    )

    # DreamPrompting (OpenAI-compatible unified gateway for Groq, Mistral, NVIDIA, Cohere, Google, OpenRouter, Cloudflare, HuggingFace, Codestral, Chat)
    # Get a key at https://dreamprompting.com/account — required for catalog and chat.
    dreamprompting_api_key: str = Field(default="", alias="DREAMPROMPTING_API_KEY")
    dreamprompting_base_url: str = Field(
        default="https://dreamprompting.com/api/v1", alias="DREAMPROMPTING_BASE_URL"
    )
    dreamprompting_catalog_refresh_seconds: int = Field(
        default=300, alias="DREAMPROMPTING_CATALOG_REFRESH_SECONDS"
    )

    # Clean APIs (OpenAI-compatible gateway, 33 models from Anthropic, OpenAI, Google, Meta)
    # Get a key at https://cleanapis.com/register — required for catalog and chat.
    cleanapis_api_key: str = Field(default="", alias="CLEANAPIS_API_KEY")
    cleanapis_base_url: str = Field(
        default="https://cleanapis.com/v1", alias="CLEANAPIS_BASE_URL"
    )
    cleanapis_catalog_refresh_seconds: int = Field(
        default=300, alias="CLEANAPIS_CATALOG_REFRESH_SECONDS"
    )

    # ModelRouter (OpenAI-compatible gateway; free models only)
    modelrouter_api_key: str = Field(default="", alias="MODELROUTER_API_KEY")
    # Kill switch: false removes ModelRouter from routing/catalogs (account
    # exhausted its 50-request free tier: 403 free_tier_limit_exceeded).
    modelrouter_enabled: bool = Field(default=False, alias="MODELROUTER_ENABLED")
    modelrouter_base_url: str = Field(
        default="https://modelrouter.id/v1", alias="MODELROUTER_BASE_URL"
    )
    modelrouter_catalog_refresh_seconds: int = Field(
        default=300, alias="MODELROUTER_CATALOG_REFRESH_SECONDS"
    )

    # Tiarina Cloud (OpenAI-compatible model gateway)
    tiarina_api_key: str = Field(default="", alias="TIARINA_API_KEY")
    # Kill switch: false removes Tiarina from routing/catalogs (account is
    # billing-locked: 403 free_tier_locked / 402 insufficient_quota).
    tiarina_enabled: bool = Field(default=False, alias="TIARINA_ENABLED")
    tiarina_base_url: str = Field(
        default="https://ai.tiarina.cloud/v1", alias="TIARINA_BASE_URL"
    )
    tiarina_catalog_refresh_seconds: int = Field(
        default=300, alias="TIARINA_CATALOG_REFRESH_SECONDS"
    )

    # ZyloAI (OpenAI-compatible model gateway; catalog is public, chat needs a key)
    zyloai_api_key: str = Field(default="", alias="ZYLOAI_API_KEY")
    zyloai_base_url: str = Field(
        default="https://api.zyloai.net/v1", alias="ZYLOAI_BASE_URL"
    )

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # Agentic reflection (critic repair pass)
    reflection_enabled: bool = Field(default=True, alias="REFLECTION_ENABLED")
    reflection_threshold: float = Field(default=0.55, alias="REFLECTION_THRESHOLD")

    # Tool result cache
    tool_cache_enabled: bool = Field(default=True, alias="TOOL_CACHE_ENABLED")
    tool_cache_ttl: int = Field(default=300, alias="TOOL_CACHE_TTL")

    # OpenTelemetry tracing (Phase 3)
    otel_exporter_otlp_endpoint: str = Field(default="", alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str = Field(default="bridgerouteai", alias="OTEL_SERVICE_NAME")

    # Capability probing (live tool support verification)
    capability_probe_enabled: bool = Field(default=True, alias="CAPABILITY_PROBE_ENABLED")
    capability_probe_ttl_seconds: int = Field(default=300, alias="CAPABILITY_PROBE_TTL_SECONDS")
    capability_probe_refresh_seconds: int = Field(default=300, alias="CAPABILITY_PROBE_REFRESH_SECONDS")

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        populate_by_name = True
        extra = "ignore"


settings = Settings()
