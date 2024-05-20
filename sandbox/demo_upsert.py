from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

import asyncio

from fdllmret.helpers.encoding import DocsetEncoding
from fdllmret.helpers.upsert import upsert_docenc

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(override=True)

docenc = DocsetEncoding.from_config(ROOT / "data_config.yml")

asyncio.run(upsert_docenc(docenc, skip_embeddings=True))