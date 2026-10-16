"""Launcher entrypoint for SpecRAG server."""

import uvicorn
from app.config import get_settings


def main():
    settings = get_settings()
    print(f"Starting {settings.APP_NAME} server on 127.0.0.1:8000...")
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )


if __name__ == "__main__":
    main()
