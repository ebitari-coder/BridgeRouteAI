# BridgeRouteAI — Credentials

## Environment Variables

Set these in your `.env` file before starting:

| Variable | Description | Example |
|---|---|---|
| `ADMIN_API_KEY` | API key for admin endpoints | `openssl rand -hex 32` |
| `JWT_SECRET` | JWT signing secret | `openssl rand -hex 32` |
| `SUPERADMIN_PASSWORD` | Admin dashboard password | `change-me-to-a-strong-password ` |
| `SUPERADMIN_EMAIL` | Admin email (default) | `admin@bridgeroute.local` |
| `CORS_ORIGINS` | Allowed frontend origins | `http://localhost:3000` |

## Quick Start

```bash
# Generate secure secrets
echo "ADMIN_API_KEY=$(openssl rand -hex 32)"
echo "JWT_SECRET=$(openssl rand -hex 32)"

# Add to .env
echo 'SUPERADMIN_PASSWORD=YourStr0ngP@ssword' >> .env

# Start services
docker compose up --build
```

## Change Password After First Login

```bash
# Login to get JWT token
curl -X POST http://localhost:8001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@bridgeroute.local", "password": "YourStr0ngP@ssword"}'

# Change password
curl -X POST http://localhost:8001/auth/change-password \
  -H "Authorization: Bearer <jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{"current_password": "YourStr0ngP@ssword", "new_password": "NewStr0ngP@ssword"}'
```

## Security Notes

- Never commit `.env` to version control
- Use strong, unique passwords (min 12 chars, mixed case + numbers + symbols)
- Rotate `ADMIN_API_KEY` and `JWT_SECRET` periodically
- Default `SUPERADMIN_EMAIL` is `admin@bridgeroute.local` unless overridden
