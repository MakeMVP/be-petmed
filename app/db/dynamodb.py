"""DynamoDB single-table design client and operations."""

from datetime import UTC, datetime
from typing import Any, TypeVar

import aioboto3
from boto3.dynamodb.conditions import Attr, Key

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class DynamoDBClient:
    """Async DynamoDB client with single-table design patterns."""

    def __init__(self) -> None:
        self._session = aioboto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        self._table_name = settings.dynamodb_table_name

    async def _get_table(self) -> Any:
        """Get DynamoDB table resource."""
        async with self._session.resource(
            "dynamodb",
            region_name=settings.aws_region,
            endpoint_url=settings.dynamodb_endpoint_url,
        ) as dynamodb:
            return await dynamodb.Table(self._table_name)

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
        async with self._session.resource(
            "dynamodb",
            region_name=settings.aws_region,
            endpoint_url=settings.dynamodb_endpoint_url,
        ) as dynamodb:
            table = await dynamodb.Table(self._table_name)

            params: dict[str, Any] = {"Item": item}
            if condition_expression:
                params["ConditionExpression"] = condition_expression

            response = await table.put_item(**params)
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
        async with self._session.resource(
            "dynamodb",
            region_name=settings.aws_region,
            endpoint_url=settings.dynamodb_endpoint_url,
        ) as dynamodb:
            table = await dynamodb.Table(self._table_name)

            response = await table.get_item(Key={"PK": pk, "SK": sk})
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

        Returns:
            Tuple of (items, last_evaluated_key for pagination).
        """
        async with self._session.resource(
            "dynamodb",
            region_name=settings.aws_region,
            endpoint_url=settings.dynamodb_endpoint_url,
        ) as dynamodb:
            table = await dynamodb.Table(self._table_name)

            # Build key condition
            key_attr = "GSI1PK" if index_name == "GSI1" else "PK"
            sk_attr = "GSI1SK" if index_name == "GSI1" else "SK"

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

            response = await table.query(**params)

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
        async with self._session.resource(
            "dynamodb",
            region_name=settings.aws_region,
            endpoint_url=settings.dynamodb_endpoint_url,
        ) as dynamodb:
            table = await dynamodb.Table(self._table_name)

            # Build update expression
            update_parts = []
            expression_names = {}
            expression_values = {}

            for i, (key, value) in enumerate(updates.items()):
                attr_name = f"#attr{i}"
                attr_value = f":val{i}"
                update_parts.append(f"{attr_name} = {attr_value}")
                expression_names[attr_name] = key
                expression_values[attr_value] = value

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

            response = await table.update_item(**params)
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
        async with self._session.resource(
            "dynamodb",
            region_name=settings.aws_region,
            endpoint_url=settings.dynamodb_endpoint_url,
        ) as dynamodb:
            table = await dynamodb.Table(self._table_name)

            params: dict[str, Any] = {
                "Key": {"PK": pk, "SK": sk},
                "ReturnValues": "ALL_OLD",
            }

            if condition_expression:
                params["ConditionExpression"] = condition_expression

            response = await table.delete_item(**params)
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
        async with self._session.resource(
            "dynamodb",
            region_name=settings.aws_region,
            endpoint_url=settings.dynamodb_endpoint_url,
        ) as dynamodb:
            table = await dynamodb.Table(self._table_name)

            params: dict[str, Any] = {}

            if filter_expression:
                params["FilterExpression"] = filter_expression
            if limit:
                params["Limit"] = limit
            if exclusive_start_key:
                params["ExclusiveStartKey"] = exclusive_start_key

            response = await table.scan(**params)

            return (
                response.get("Items", []),
                response.get("LastEvaluatedKey"),
            )

    async def batch_write(self, items: list[dict[str, Any]]) -> None:
        """Batch write items (put or delete).

        Args:
            items: List of items to write. Each item should have PK and SK.
        """
        async with self._session.resource(
            "dynamodb",
            region_name=settings.aws_region,
            endpoint_url=settings.dynamodb_endpoint_url,
        ) as dynamodb:
            table = await dynamodb.Table(self._table_name)

            async with table.batch_writer() as batch:
                for item in items:
                    await batch.put_item(Item=item)

            logger.debug("Batch wrote items", count=len(items))

    async def batch_delete(self, keys: list[tuple[str, str]]) -> None:
        """Batch delete items by keys.

        Args:
            keys: List of (pk, sk) tuples to delete.
        """
        async with self._session.resource(
            "dynamodb",
            region_name=settings.aws_region,
            endpoint_url=settings.dynamodb_endpoint_url,
        ) as dynamodb:
            table = await dynamodb.Table(self._table_name)

            async with table.batch_writer() as batch:
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
        async with self._session.client(
            "dynamodb",
            region_name=settings.aws_region,
            endpoint_url=settings.dynamodb_endpoint_url,
        ) as client:
            await client.transact_write_items(TransactItems=items)
            logger.debug("Transaction write completed", item_count=len(items))


# Global client instance
_db_client: DynamoDBClient | None = None


def get_dynamodb_client() -> DynamoDBClient:
    """Get or create the DynamoDB client singleton."""
    global _db_client
    if _db_client is None:
        _db_client = DynamoDBClient()
    return _db_client
