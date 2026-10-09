#!/usr/bin/env python3
"""Start the AUDI FastAPI backend on the host/port from config.yaml (`api:`)."""

import logging

import uvicorn

from ml.config import cfg, load_config

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config()
    uvicorn.run("api.main:app", host=cfg(config, "api", "host"), port=int(cfg(config, "api", "port")))
