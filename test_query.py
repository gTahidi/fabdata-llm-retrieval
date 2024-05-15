# %%
from __future__ import annotations

import json
import argparse
import asyncio

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from fdllm import get_caller
from fdllm.sysutils import register_models
from fdllm.chat import ChatController
from fdllmret.datastore.factory import get_datastore
from fdllmret.plugin import RetrievalPlugin

# %%

HERE = Path(__file__).resolve().parent


async def querywrapper(chatcontroller: ChatController):
    query = input("Prompt: ")
    if query == "exit":
        return False
    _, response = await chatcontroller.achat(query)
    print(f"\n---------\n{chatcontroller.recent_tool_calls}\n---------\n\n")
    print(f"\n---------\n{response.Message}\n---------\n\n")
    return True


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default=str(HERE / "context.txt"))
    parser.add_argument("--model", default="fabdata-openai-eastus2-gpt4turbo")
    parser.add_argument("--json-db", required=True, type=str)
    parser.add_argument("--contents-db", required=True, type=str)
    args = parser.parse_args()

    BASE = Path(__file__).parents[1]

    with open(args.context, "r") as f:
        ctx = "\n".join(ln for ln in f if not ln.strip() or ln.strip()[0] != "#")
    datastore = await get_datastore()
    register_models(Path.home() / ".fdllm/custom_models.yaml")

    with open(args.json_db) as f:
        json_db = json.load(f)
    with open(args.contents_db) as f:
        contents_db = json.load(f)

    caller = get_caller(args.model)
    chatcontroller = ChatController(
        Caller=caller,
        Sys_Msg={0: ctx, -1: ctx},
        Keep_History=True,
    )
    plugin = RetrievalPlugin(
        datastore=datastore,
        json_contents=contents_db,
        json_database=json_db,
        tags=["guides", "research", "reviews"],
        supp_tags=["supporting material"],
        chunksizes=[200, 400, 600, 800, 1000],
    )
    chatcontroller.register_plugin(plugin)
    while await querywrapper(chatcontroller):
        pass
    await datastore.client.connection_pool.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
