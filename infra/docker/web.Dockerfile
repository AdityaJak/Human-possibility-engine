# syntax=docker/dockerfile:1.7
# Human Possibility Engine web app. Build from the repo root:
#   docker build -f infra/docker/web.Dockerfile --build-arg VITE_API_BASE_URL=http://localhost:8000 -t hpe-web:0.1.0 .
FROM node:22-slim AS build
WORKDIR /web
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY apps/web ./
# The API base URL is baked in at build time (Vite); the browser calls the API directly.
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

# Static serving with unprivileged nginx (listens on 8080, runs as non-root).
FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime
COPY infra/docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
  CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1:8080/"]
