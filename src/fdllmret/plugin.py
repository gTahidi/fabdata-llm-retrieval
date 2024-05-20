from fdllm.tooluse import ToolUsePlugin
from typing import Optional, Union, List
import os

from .datastore.factory import get_datastore
from .tools import *
from .helpers.encoding import DocsetEncoding


async def retrieval_plugin(
    dbhost: Optional[str] = None,
    dbport: Optional[str] = None,
    dbssl: Optional[str] = None,
    chunksizes: Optional[Union[int, List[int]]] = None,
):
    if dbhost is not None:
        os.environ["REDIS_HOST"] = dbhost
    if dbport is not None:
        os.environ["REDIS_PORT"] = dbport
    if dbssl is not None:
        os.environ["REDIS_SSL"] = dbssl

    datastore = await get_datastore()
    docenc = await DocsetEncoding.from_datastore(datastore)

    if chunksizes is None:
        chunksizes = docenc.chunk_sizes
    else:
        if isinstance(chunksizes, int):
            chunksizes = [chunksizes]
        if not set(chunksizes).issubset(set(docenc.chunk_sizes)):
            raise ValueError("chunksize must be a subset of docenc.docembs.chunk_sizes")

    plugin = RetrievalPlugin(
        datastore=datastore,
        json_contents=docenc.contents,
        json_database=docenc.jsondata,
        chunksizes=chunksizes,
        tags=docenc.tags,
        supp_tags=docenc.supp_tags,
    )
    
    return plugin, datastore


class RetrievalPlugin(ToolUsePlugin):
    def __init__(
        self, datastore, json_contents, json_database, chunksizes, tags, supp_tags
    ):
        tools = [
            QueryCatalogue(datastore=datastore, tags=tags, chunksizes=chunksizes),
            GetReferences(json_database=json_database),
        ]
        if json_contents:
            tools.append(GetContents(json_contents=json_contents))
        if supp_tags:
            tools.append(
                QuerySuppMat(
                    datastore=datastore,
                    json_database=json_database,
                    tags=supp_tags,
                    chunksizes=chunksizes[-1:],
                ),
            )
        super().__init__(Tools=tools)
