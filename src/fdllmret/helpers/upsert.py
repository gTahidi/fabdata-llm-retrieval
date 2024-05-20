from tqdm import tqdm

from ..datastore.factory import get_datastore
from .encoding import DocsetEncoding

async def upsert_docenc(docenc: DocsetEncoding, batch_size=5, skip_embeddings=False):
    datastore = await get_datastore()
    
    if not skip_embeddings:
        batchdocs = []
        batchchunks = {}
        for doc, chunks in tqdm(docenc):
            batchdocs.append(doc)
            batchchunks[doc.id] = chunks[doc.id]
            if len(batchdocs) == batch_size:
                await datastore.upsert(documents=batchdocs, chunks=batchchunks)
                batchdocs = []
                batchchunks = {}
                
    await datastore.client.json().set("fulldb", "$", docenc.jsondata)
    await datastore.client.json().set("contents", "$", docenc.contents)
    await datastore.client.json().set("config", "$", docenc.config)
    
    await datastore.client.connection_pool.disconnect()