"""DynamoDB single-table design client and operations."""

import asyncio
import functools
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import Any, TypeVar

from boto3.dynamodb.conditions import Key
from boto3.dynamodb.types import TypeSerializer

from app.config import settings
from app.core.aws import get_aioboto3_session
from app.core.logging import get_logger
from app.db.types import sanitize_floats

logger = get_logger(__name__)

T = TypeVar("T")
_serializer = TypeSerializer()


class DynamoDBClient:
    """Async DynamoDB client with single-table design patterns."""

    def __init__(self) -> None:
        self._session = get_aioboto3_session()
        self._table_name = settings.dynamodb_table_name
        self._table: Any = None
        self._dynamodb: Any = None
        self._client: Any = None
        self._exit_stack: AsyncExitStack | None = None

    async def connect(self) -> None:
        """Open a long-lived DynamoDB resource and table reference."""
        if self._table is not None:
            return  # Already connected
        self._exit_stack = AsyncExitStack()
        self._dynamodb = await self._exit_stack.enter_async_context(
            self._session.resource(
                "dynamodb",
                region_name=settings.aws_region,
                endpoint_url=settings.dynamodb_endpoint_url,
            )
        )
        self._table = await self._dynamodb.Table(self._table_name)
        self._client = await self._exit_stack.enter_async_context(
            self._session.client(
                "dynamodb",
                region_name=settings.aws_region,
                endpoint_url=settings.dynamodb_endpoint_url,
            )
        )
        logger.info("DynamoDB connection pool opened", table=self._table_name)

    async def close(self) -> None:
        """Tear down the connection pool."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._table = None
            self._dynamodb = None
            self._client = None
            logger.info("DynamoDB connection pool closed")

    @property
    def table(self) -> Any:
        """Return the long-lived table reference."""
        if self._table is None:
            raise RuntimeError("DynamoDBClient not connected — call connect() first")
        return self._table

    @staticmethod
    def _key_attrs(index_name: str | None) -> tuple[str, str]:
        """Return (partition_key_attr, sort_key_attr) for the given index."""
        if index_name == "GSI1":
            return "GSI1PK", "GSI1SK"
        if index_name == "GSI2":
            return "GSI2PK", "GSI2SK"
        return "PK", "SK"

    async def put_item(
        self,
        item: dict[str, Any],
        condition_expression: Any | None = None,
    ) -> dict[str, Any]:
        """Put an item into DynamoDB.

        Args:
            item: The item to store.
            condition_expression: Optional condition for the put.

        Returns:
            The response from DynamoDB.
        """
        params: dict[str, Any] = {"Item": sanitize_floats(item)}
        if condition_expression:
            params["ConditionExpression"] = condition_expression

        response = await self.table.put_item(**params)
        logger.debug("Put item", pk=item.get("PK"), sk=item.get("SK"))
        return response

    async def get_item(self, pk: str, sk: str) -> dict[str, Any] | None:
        """Get a single item by primary key.

        Args:
            pk: Partition key value.
            sk: Sort key value.

        Returns:
            The item if found, None otherwise.
        """
        response = await self.table.get_item(Key={"PK": pk, "SK": sk})
        return response.get("Item")

    async def query(
        self,
        pk: str,
        sk_prefix: str | None = None,
        sk_begins_with: str | None = None,
        index_name: str | None = None,
        limit: int | None = None,
        scan_forward: bool = True,
        exclusive_start_key: dict[str, Any] | None = None,
        filter_expression: Any | None = None,
        projection_expression: str | None = None,
        expression_attribute_names: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Query items by partition key with optional sort key conditions.

        Args:
            pk: Partition key value.
            sk_prefix: Sort key exact prefix match.
            sk_begins_with: Sort key begins_with condition.
            index_name: Optional GSI name to query.
            limit: Maximum items to return.
            scan_forward: True for ascending, False for descending.
            exclusive_start_key: Pagination cursor.
            filter_expression: Optional filter expression.
            projection_expression: Optional projection expression.
            expression_attribute_names: Optional expression attribute names.

        Returns:
            Tuple of (items, last_evaluated_key for pagination).
        """
        # Build key condition
        key_attr, sk_attr = self._key_attrs(index_name)
        key_condition = Key(key_attr).eq(pk)

        if sk_prefix:
            key_condition = key_condition & Key(sk_attr).eq(sk_prefix)
        elif sk_begins_with:
            key_condition = key_condition & Key(sk_attr).begins_with(sk_begins_with)

        # Build query params
        params: dict[str, Any] = {
            "KeyConditionExpression": key_condition,
            "ScanIndexForward": scan_forward,
        }

        if index_name:
            params["IndexName"] = index_name
        if limit:
            params["Limit"] = limit
        if exclusive_start_key:
            params["ExclusiveStartKey"] = exclusive_start_key
        if filter_expression:
            params["FilterExpression"] = filter_expression
        if projection_expression:
            params["ProjectionExpression"] = projection_expression
        if expression_attribute_names:
            params["ExpressionAttributeNames"] = expression_attribute_names

        response = await self.table.query(**params)

        return (
            response.get("Items", []),
            response.get("LastEvaluatedKey"),
        )

    async def update_item(
        self,
        pk: str,
        sk: str,
        updates: dict[str, Any],
        condition_expression: Any | None = None,
    ) -> dict[str, Any]:
        """Update an item's attributes.

        Args:
            pk: Partition key value.
            sk: Sort key value.
            updates: Dictionary of attribute updates.
            condition_expression: Optional condition for the update.

        Returns:
            The updated item.
        """
        # Build update expression
        update_parts = []
        expression_names = {}
        expression_values = {}

        for i, (key, value) in enumerate(updates.items()):
            attr_name = f"#attr{i}"
            attr_value = f":val{i}"
            update_parts.append(f"{attr_name} = {attr_value}")
            expression_names[attr_name] = key
            expression_values[attr_value] = sanitize_floats(value)

        # Always update updated_at timestamp
        update_parts.append("#updated_at = :updated_at")
        expression_names["#updated_at"] = "updated_at"
        expression_values[":updated_at"] = datetime.now(UTC).isoformat()

        update_expression = "SET " + ", ".join(update_parts)

        params: dict[str, Any] = {
            "Key": {"PK": pk, "SK": sk},
            "UpdateExpression": update_expression,
            "ExpressionAttributeNames": expression_names,
            "ExpressionAttributeValues": expression_values,
            "ReturnValues": "ALL_NEW",
        }

        if condition_expression:
            params["ConditionExpression"] = condition_expression

        response = await self.table.update_item(**params)
        logger.debug("Updated item", pk=pk, sk=sk)
        return response.get("Attributes", {})

    async def delete_item(
        self,
        pk: str,
        sk: str,
        condition_expression: Any | None = None,
    ) -> bool:
        """Delete an item.

        Args:
            pk: Partition key value.
            sk: Sort key value.
            condition_expression: Optional condition for the delete.

        Returns:
            True if deleted, False otherwise.
        """
        params: dict[str, Any] = {
            "Key": {"PK": pk, "SK": sk},
            "ReturnValues": "ALL_OLD",
        }

        if condition_expression:
            params["ConditionExpression"] = condition_expression

        response = await self.table.delete_item(**params)
        deleted = "Attributes" in response
        logger.debug("Deleted item", pk=pk, sk=sk, deleted=deleted)
        return deleted

    async def scan(
        self,
        filter_expression: Any | None = None,
        limit: int | None = None,
        exclusive_start_key: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Scan the entire table with optional filtering.

        Args:
            filter_expression: Optional filter expression.
            limit: Maximum items to return.
            exclusive_start_key: Pagination cursor.

        Returns:
            Tuple of (items, last_evaluated_key for pagination).
        """
        params: dict[str, Any] = {}

        if filter_expression:
            params["FilterExpression"] = filter_expression
        if limit:
            params["Limit"] = limit
        if exclusive_start_key:
            params["ExclusiveStartKey"] = exclusive_start_key

        response = await self.table.scan(**params)

        return (
            response.get("Items", []),
            response.get("LastEvaluatedKey"),
        )

    async def increment_counter(
        self,
        pk: str,
        sk: str,
        counter_attr: str,
        increment: int = 1,
        set_updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically increment a numeric attribute using ADD expression.

        Args:
            pk: Partition key value.
            sk: Sort key value.
            counter_attr: The attribute to increment.
            increment: Amount to add (can be negative to decrement).
            set_updates: Optional additional SET updates to apply atomically.

        Returns:
            The updated item.
        """
        return await self.increment_counters(
            pk=pk, sk=sk, counters={counter_attr: increment}, set_updates=set_updates,
        )

    async def increment_counters(
        self,
        pk: str,
        sk: str,
        counters: dict[str, int],
        set_updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically increment multiple numeric attributes using ADD expression.

        Args:
            pk: Partition key value.
            sk: Sort key value.
            counters: Dict of {attribute_name: increment_value}.
            set_updates: Optional additional SET updates to apply atomically.

        Returns:
            The updated item.
        """
        expression_names: dict[str, str] = {}
        expression_values: dict[str, Any] = {}

        add_parts = []
        for i, (attr, inc) in enumerate(counters.items()):
            name_key = f"#cnt{i}"
            val_key = f":cinc{i}"
            add_parts.append(f"{name_key} {val_key}")
            expression_names[name_key] = attr
            expression_values[val_key] = inc

        add_clause = "ADD " + ", ".join(add_parts)

        # Always update updated_at + any extra SET updates
        set_parts = ["#updated_at = :updated_at"]
        expression_names["#updated_at"] = "updated_at"
        expression_values[":updated_at"] = datetime.now(UTC).isoformat()

        if set_updates:
            for i, (key, value) in enumerate(set_updates.items()):
                attr_name = f"#sattr{i}"
                attr_value = f":sval{i}"
                set_parts.append(f"{attr_name} = {attr_value}")
                expression_names[attr_name] = key
                expression_values[attr_value] = value

        update_expression = f"{add_clause} SET {', '.join(set_parts)}"

        response = await self.table.update_item(
            Key={"PK": pk, "SK": sk},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_names,
            ExpressionAttributeValues=expression_values,
            ReturnValues="ALL_NEW",
        )
        logger.debug("Incremented counters", pk=pk, sk=sk, attrs=list(counters.keys()))
        return response.get("Attributes", {})

    async def transact_put_and_increment(
        self,
        item: dict[str, Any],
        counter_pk: str,
        counter_sk: str,
        counters: dict[str, int],
    ) -> None:
        """Atomically put an item and increment counters in a single transaction.

        Uses TransactWriteItems to ensure both the put and the counter update
        succeed or both fail (prevents counter drift).

        Args:
            item: The item to put (high-level format, not typed).
            counter_pk: PK of the counter item to update.
            counter_sk: SK of the counter item to update.
            counters: Dict of {attribute_name: increment_value}.
        """
        serialized_item = {k: _serializer.serialize(v) for k, v in item.items()}

        expression_names: dict[str, str] = {}
        expression_values: dict[str, Any] = {}
        add_parts = []
        for i, (attr, inc) in enumerate(counters.items()):
            name_key = f"#cnt{i}"
            val_key = f":cinc{i}"
            add_parts.append(f"{name_key} {val_key}")
            expression_names[name_key] = attr
            expression_values[val_key] = _serializer.serialize(inc)

        set_parts = ["#ua = :ua"]
        expression_names["#ua"] = "updated_at"
        expression_values[":ua"] = _serializer.serialize(datetime.now(UTC).isoformat())

        update_expression = f"ADD {', '.join(add_parts)} SET {', '.join(set_parts)}"

        await self._client.transact_write_items(
            TransactItems=[
                {
                    "Put": {
                        "TableName": self._table_name,
                        "Item": serialized_item,
                    }
                },
                {
                    "Update": {
                        "TableName": self._table_name,
                        "Key": {
                            "PK": _serializer.serialize(counter_pk),
                            "SK": _serializer.serialize(counter_sk),
                        },
                        "UpdateExpression": update_expression,
                        "ExpressionAttributeNames": expression_names,
                        "ExpressionAttributeValues": expression_values,
                    }
                },
            ]
        )
        logger.debug(
            "Transact put + increment",
            pk=item.get("PK"),
            counters=list(counters.keys()),
        )

    async def query_across_shards(
        self,
        shard_pks: list[str],
        index_name: str,
        limit: int,
        cursor: dict | None = None,
        scan_forward: bool = False,
    ) -> tuple[list[dict[str, Any]], dict | None]:
        """Scatter-gather query across multiple shard partition keys.

        Queries all non-exhausted shards in parallel, merges results by sort key,
        and returns top ``limit`` items with a compound cursor for pagination.

        Args:
            shard_pks: List of partition key values (one per shard).
            index_name: GSI name to query.
            limit: Maximum items to return.
            cursor: Compound cursor from previous call (``{"s": {...}}``).
            scan_forward: True for ascending, False for descending.

        Returns:
            Tuple of (items, next_compound_cursor or None).
        """
        shard_state = cursor.get("s", {}) if cursor else {}
        pk_attr, sk_attr = self._key_attrs(index_name)

        # Query non-exhausted shards in parallel
        tasks = []
        active_shards: list[str] = []
        for shard_pk in shard_pks:
            state = shard_state.get(shard_pk, {})
            if state.get("d"):
                continue
            active_shards.append(shard_pk)
            tasks.append(
                self.query(
                    pk=shard_pk,
                    index_name=index_name,
                    limit=limit,
                    exclusive_start_key=state.get("k"),
                    scan_forward=scan_forward,
                )
            )

        if not tasks:
            return [], None

        results = await asyncio.gather(*tasks)

        # Collect items tagged with their shard
        tagged: list[tuple[dict, str]] = []
        shard_meta: dict[str, tuple[list[dict], dict | None]] = {}
        for shard_pk, (items, last_key) in zip(active_shards, results):
            shard_meta[shard_pk] = (items, last_key)
            for item in items:
                tagged.append((item, shard_pk))

        # Merge-sort by SK attribute
        tagged.sort(
            key=lambda x: x[0].get(sk_attr, ""),
            reverse=not scan_forward,
        )

        # Take top `limit`
        selected = tagged[:limit]
        selected_items = [item for item, _ in selected]

        # Track how many items from each shard were selected
        used_per_shard: dict[str, int] = {}
        last_used_item: dict[str, dict] = {}
        for item, shard_pk in selected:
            used_per_shard[shard_pk] = used_per_shard.get(shard_pk, 0) + 1
            last_used_item[shard_pk] = item

        # Build next compound cursor
        next_state: dict[str, dict] = {}
        any_remaining = False

        for shard_pk in shard_pks:
            old = shard_state.get(shard_pk, {})
            if old.get("d"):
                next_state[shard_pk] = {"d": True}
                continue

            if shard_pk not in shard_meta:
                continue

            items_returned, last_key = shard_meta[shard_pk]
            n_used = used_per_shard.get(shard_pk, 0)

            if n_used == 0:
                # Shard had items but none made the cut — keep current cursor
                next_state[shard_pk] = old
                if items_returned:
                    any_remaining = True
            elif n_used < len(items_returned):
                # Partially used — resume from last included item
                li = last_used_item[shard_pk]
                next_state[shard_pk] = {
                    "k": {
                        pk_attr: li[pk_attr],
                        sk_attr: li[sk_attr],
                        "PK": li["PK"],
                        "SK": li["SK"],
                    }
                }
                any_remaining = True
            elif last_key:
                # Fully used, more items exist
                next_state[shard_pk] = {"k": last_key}
                any_remaining = True
            else:
                # Fully used, shard exhausted
                next_state[shard_pk] = {"d": True}

        if not any_remaining:
            return selected_items, None

        return selected_items, {"s": next_state}

    async def batch_get_items(
        self,
        keys: list[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        """Batch get items by keys (up to 100 at a time).

        Args:
            keys: List of (pk, sk) tuples to fetch.

        Returns:
            List of items found.
        """
        if not keys:
            return []

        results: list[dict[str, Any]] = []

        # DynamoDB batch_get_item supports max 100 keys per request
        for i in range(0, len(keys), 100):
            batch_keys = keys[i : i + 100]

            response = await self._dynamodb.batch_get_item(
                RequestItems={
                    self._table_name: {
                        "Keys": [
                            {"PK": pk, "SK": sk} for pk, sk in batch_keys
                        ]
                    }
                }
            )
            results.extend(response.get("Responses", {}).get(self._table_name, []))

            # Handle unprocessed keys with exponential backoff
            unprocessed = response.get("UnprocessedKeys", {})
            delay = 0.1
            max_retries = 5
            retries = 0
            while unprocessed.get(self._table_name) and retries < max_retries:
                await asyncio.sleep(delay)
                retry_response = await self._dynamodb.batch_get_item(
                    RequestItems=unprocessed
                )
                results.extend(
                    retry_response.get("Responses", {}).get(self._table_name, [])
                )
                unprocessed = retry_response.get("UnprocessedKeys", {})
                retries += 1
                delay = min(delay * 2, 5.0)

            if unprocessed.get(self._table_name):
                remaining = len(unprocessed[self._table_name].get("Keys", []))
                logger.warning(
                    "Unprocessed keys remain after retries",
                    remaining=remaining,
                    retries=max_retries,
                )

        logger.debug("Batch get items", requested=len(keys), found=len(results))
        return results

    async def query_count(
        self,
        pk: str,
        sk_begins_with: str | None = None,
        index_name: str | None = None,
    ) -> int:
        """Count items matching a query without fetching them.

        Args:
            pk: Partition key value.
            sk_begins_with: Sort key begins_with condition.
            index_name: Optional GSI name.

        Returns:
            Number of matching items.
        """
        key_attr, sk_attr = self._key_attrs(index_name)
        key_condition = Key(key_attr).eq(pk)
        if sk_begins_with:
            key_condition = key_condition & Key(sk_attr).begins_with(sk_begins_with)

        params: dict[str, Any] = {
            "KeyConditionExpression": key_condition,
            "Select": "COUNT",
        }
        if index_name:
            params["IndexName"] = index_name

        total = 0
        while True:
            response = await self.table.query(**params)
            total += response.get("Count", 0)
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            params["ExclusiveStartKey"] = last_key

        return total

    async def batch_write(self, items: list[dict[str, Any]]) -> None:
        """Batch write items (put or delete).

        Args:
            items: List of items to write. Each item should have PK and SK.
        """
        async with self.table.batch_writer() as batch:
            for item in items:
                await batch.put_item(Item=item)

        logger.debug("Batch wrote items", count=len(items))

    async def batch_delete(self, keys: list[tuple[str, str]]) -> None:
        """Batch delete items by keys.

        Args:
            keys: List of (pk, sk) tuples to delete.
        """
        async with self.table.batch_writer() as batch:
            for pk, sk in keys:
                await batch.delete_item(Key={"PK": pk, "SK": sk})

        logger.debug("Batch deleted items", count=len(keys))

    async def transact_write(
        self,
        items: list[dict[str, Any]],
    ) -> None:
        """Transactional write of multiple items.

        Args:
            items: List of transact items (Put, Update, Delete, ConditionCheck).
        """
        await self._client.transact_write_items(TransactItems=items)
        logger.debug("Transaction write completed", item_count=len(items))

    async def query_all_keys(
        self,
        pk: str,
        sk_begins_with: str | None = None,
        index_name: str | None = None,
    ) -> list[tuple[str, str]]:
        """Query all primary keys matching a pattern, paginating through all results.

        Uses ProjectionExpression to minimize RCU cost.

        Args:
            pk: Partition key value.
            sk_begins_with: Sort key begins_with condition.
            index_name: Optional GSI name.

        Returns:
            List of (PK, SK) tuples.
        """
        key_attr, sk_attr = self._key_attrs(index_name)
        key_condition = Key(key_attr).eq(pk)
        if sk_begins_with:
            key_condition = key_condition & Key(sk_attr).begins_with(sk_begins_with)

        params: dict[str, Any] = {
            "KeyConditionExpression": key_condition,
            "ProjectionExpression": "PK, SK",
        }
        if index_name:
            params["IndexName"] = index_name

        all_keys: list[tuple[str, str]] = []
        while True:
            response = await self.table.query(**params)
            for item in response.get("Items", []):
                all_keys.append((item["PK"], item["SK"]))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            params["ExclusiveStartKey"] = last_key

        return all_keys


@functools.lru_cache
def get_dynamodb_client() -> DynamoDBClient:
    """Get or create the DynamoDB client singleton."""
    return DynamoDBClient()
