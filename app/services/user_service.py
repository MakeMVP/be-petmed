"""User management service — shared business logic."""

import asyncio

from app.core.logging import get_logger
from app.db.dynamodb import DynamoDBClient
from app.services.pinecone_service import PineconeService
from app.services.storage_service import StorageService

logger = get_logger(__name__)


async def delete_user_data(
    user_id: str,
    db: DynamoDBClient,
    storage: StorageService,
    vector_db: PineconeService,
) -> int:
    """Delete all data associated with a user.

    Deletes S3 files, Pinecone vectors, and all DynamoDB items
    (user record, documents, chunks, conversations, messages, queries).
    Uses paginated queries to handle >1MB of data.

    Args:
        user_id: The user ID whose data should be deleted.
        db: DynamoDB client.
        storage: S3 storage service.
        vector_db: Pinecone vector database.

    Returns:
        Number of DynamoDB items deleted.
    """
    user_pk = f"USER#{user_id}"

    # Fetch doc keys and conv keys in parallel
    doc_keys, conv_keys = await asyncio.gather(
        db.query_all_keys(pk=user_pk, sk_begins_with="DOC#"),
        db.query_all_keys(pk=user_pk, sk_begins_with="CONV#"),
    )

    # Process documents: S3 deletes + chunk key lookups in parallel
    chunk_keys: list[tuple[str, str]] = []
    if doc_keys:
        doc_items = await db.batch_get_items(doc_keys)

        # Delete all S3 files in parallel
        s3_tasks = [
            storage.delete_file(doc["s3_key"])
            for doc in doc_items
            if doc.get("s3_key")
        ]

        # Query chunk keys for all documents in parallel
        chunk_tasks = [
            db.query_all_keys(pk=f"DOC#{sk.removeprefix('DOC#')}", sk_begins_with="CHUNK#")
            for _pk, sk in doc_keys
        ]

        # Run S3 deletes and chunk queries concurrently
        chunk_results = await asyncio.gather(
            asyncio.gather(*s3_tasks, return_exceptions=True),
            asyncio.gather(*chunk_tasks),
        )

        # Log any S3 deletion failures but don't abort
        for result in chunk_results[0]:
            if isinstance(result, Exception):
                logger.warning("S3 file deletion failed during user cleanup", error=str(result))

        for keys in chunk_results[1]:
            chunk_keys.extend(keys)

    # Delete all vectors for this user
    await vector_db.delete_vectors(filter_dict={"user_id": user_id})

    # Get message, query, and query-index keys for all conversations in parallel
    msg_keys: list[tuple[str, str]] = []
    query_keys: list[tuple[str, str]] = []
    query_index_keys: list[tuple[str, str]] = []

    if conv_keys:
        # Build parallel tasks for all conversations
        msg_tasks = []
        query_tasks = []
        for _pk, sk in conv_keys:
            conv_id = sk.removeprefix("CONV#")
            msg_tasks.append(db.query_all_keys(pk=f"CONV#{conv_id}", sk_begins_with="MSG#"))
            query_tasks.append(db.query_all_keys(pk=f"CONV#{conv_id}", sk_begins_with="QUERY#"))

        msg_results, query_results = await asyncio.gather(
            asyncio.gather(*msg_tasks),
            asyncio.gather(*query_tasks),
        )

        for keys in msg_results:
            msg_keys.extend(keys)

        for keys in query_results:
            query_keys.extend(keys)
            # Clean up query index items (PK=QUERY#{query_id})
            for _qpk, qsk in keys:
                query_id = qsk.removeprefix("QUERY#")
                query_index_keys.append((f"QUERY#{query_id}", f"QUERY#{query_id}"))

    # Collect all items to delete
    items_to_delete: list[tuple[str, str]] = [(user_pk, user_pk)]
    items_to_delete.extend(doc_keys)
    items_to_delete.extend(chunk_keys)
    items_to_delete.extend(conv_keys)
    items_to_delete.extend(msg_keys)
    items_to_delete.extend(query_keys)
    items_to_delete.extend(query_index_keys)

    await db.batch_delete(items_to_delete)

    logger.info(
        "Deleted user data",
        user_id=user_id,
        items_deleted=len(items_to_delete),
    )

    return len(items_to_delete)
