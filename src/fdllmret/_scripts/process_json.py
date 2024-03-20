import uuid
import json
import argparse
import asyncio
from pathlib import Path
import pickle
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv
load_dotenv(override=True)

from ..models.models import Document, DocumentMetadata, Optional
from ..datastore.datastore import DataStore
from ..datastore.factory import get_datastore
from ..services.chunks import get_document_chunks, CHUNK_SIZE


DOCUMENT_UPSERT_BATCH_SIZE = 10

async def process_json_dump(
    filepath: str,
    custom_metadata: dict,
    datastore: Optional[DataStore] = None,
    frompkl: bool = False,
    chunk_size: int = [CHUNK_SIZE],
):
    pklfile = Path(filepath).with_suffix(".pkl")
    if datastore is None:
        outdata = dict()
        
    # load the json file as a list of dictionaries
    with open(filepath) as json_file:
        data = json.load(json_file)

    documents = []
    skipped_items = []
    # iterate over the data and create document objects
    for item in data:
        if len(documents) % 20 == 0:
            print(f"Processed {len(documents)} documents")

        try:
            # get the id, text, source, source_id, url, created_at and author from the item
            # use default values if not specified
            id = item.get("id", None)
            text = item.get("text", None)
            source = item.get("source", None)
            source_id = item.get("source_id", None)
            url = item.get("url", None)
            created_at = item.get("created_at", None)
            author = item.get("author", None)
            filename = item.get("filename", None)
            tag = item.get("tag", None)

            if not text:
                print("No document text, skipping...")
                continue

            # create a metadata object with the source, source_id, url, created_at and author
            metadata = DocumentMetadata(
                source=source,
                source_id=source_id,
                url=url,
                created_at=created_at,
                author=author,
                filename=filename,
                tag=tag,
            )
            print("metadata: ", str(metadata))

            # update metadata with custom values
            for key, value in custom_metadata.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)

            # create a document object with the id or a random id, text and metadata
            document = Document(
                id=id or str(uuid.uuid4()),
                text=text,
                metadata=metadata,
            )
            documents.append(document)
        except Exception as e:
            # log the error and continue with the next item
            print(f"Error processing {item}: {e}")
            skipped_items.append(item)  # add the skipped item to the list

    # do this in batches, the upsert method already batches documents but this allows
    # us to add more descriptive logging
    if frompkl:
        with open(pklfile, "rb") as f:
            pklchunks = pickle.load(f)
    for i in range(0, len(documents), DOCUMENT_UPSERT_BATCH_SIZE):
        # Get the text of the chunks in the current batch
        batch_documents = documents[i : i + DOCUMENT_UPSERT_BATCH_SIZE]
        print(f"Upserting batch of {len(batch_documents)} documents, batch {i}")
        if datastore is not None:
            if frompkl:
                batch_chunks = {doc.id: pklchunks[doc.id] for doc in batch_documents}
                await datastore.upsert(batch_documents, chunks=batch_chunks)
            else:
                await datastore.upsert(batch_documents)
        else:
            docchunks = defaultdict(list)
            for chunk_size_ in chunk_size:
                chunks = get_document_chunks(batch_documents, chunk_size_)
                for key, val in chunks.items():
                    docchunks[key].extend(val)
            outdata = {**outdata, **docchunks}
    if datastore is None:
        with open(pklfile, "wb") as f:
            pickle.dump(outdata, f)
    # print the skipped items
    print(f"Skipped {len(skipped_items)} items due to errors or PII detection")
    for item in skipped_items:
        print(item)


async def amain():
    # parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", required=True, help="The path to the json dump")
    parser.add_argument(
        "--custom_metadata",
        default="{}",
        help="A JSON string of key-value pairs to update the metadata of the documents",
    )
    parser.add_argument(
        "--chunk_size",
        default=[CHUNK_SIZE],
        nargs="+",
        type=int,
        help="A boolean flag to indicate whether to try the PII detection function (using a language model)",
    )
    parser.add_argument(
        "--no-upsert",
        dest="upsert",
        action="store_false",
        help="A boolean flag to indicate whether to upsert to datastore or just save document chunks to pickle",
    )
    parser.add_argument(
        "--from-pkl",
        dest="frompkl",
        action="store_true",
        help="A boolean flag to indicate whether to upsert to datastore or just save document chunks to pickle",
    )
    args = parser.parse_args()

    # get the arguments
    filepath = args.filepath
    custom_metadata = json.loads(args.custom_metadata)
    screen_for_pii = args.screen_for_pii
    extract_metadata = args.extract_metadata
    chunk_size = args.chunk_size
    if not isinstance(chunk_size, list):
        chunk_size = [chunk_size]

    # initialize the db instance once as a global variable
    datastore = await get_datastore() if args.upsert else None
    # process the json dump
    await process_json_dump(
        filepath,
        custom_metadata,
        screen_for_pii,
        extract_metadata,
        datastore,
        args.frompkl,
        chunk_size,
        args.fulldoc,
    )
    if datastore is not None:
        await datastore.client.connection_pool.disconnect()

def main():
    asyncio.run(amain())

if __name__ == "__main__":
    main()
