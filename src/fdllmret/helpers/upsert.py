from tqdm import tqdm

from ..datastore.factory import get_datastore
from .encoding import DocsetEncoding

async def upsert_docenc(docenc: DocsetEncoding, batch_size=5):
    datastore = await get_datastore()
    
    batchdocs = []
    batchchunks = {}
    for doc, chunks in tqdm(docenc):
        batchdocs.append(doc)
        batchchunks[doc.id] = chunks[doc.id]
        if len(batchdocs) == batch_size:
            await datastore.upsert(documents=batchdocs, chunks=batchchunks)
            batchdocs = []
            batchchunks = {}
    
    await datastore.client.connection_pool.disconnect()