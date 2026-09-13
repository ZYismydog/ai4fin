"""Run the local AI4Fin browser workbench."""

import os

import uvicorn

from .app import create_app


def main() -> None:
    host = os.getenv("AI4FIN_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("AI4FIN_WEB_PORT", "4173"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
