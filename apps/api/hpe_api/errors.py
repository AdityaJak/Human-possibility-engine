"""Domain errors mapped to HTTP status codes."""

from __future__ import annotations


class ServiceError(Exception):
    status_code = 500

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class BadRequest(ServiceError):
    status_code = 400


class Forbidden(ServiceError):
    status_code = 403


class NotFound(ServiceError):
    status_code = 404


class Conflict(ServiceError):
    status_code = 409


class NotImplementedYet(ServiceError):
    status_code = 501


class Unavailable(ServiceError):
    status_code = 503
