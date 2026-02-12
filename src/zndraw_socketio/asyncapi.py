"""AsyncAPI 3.0.0 schema generation from handler metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel


@dataclass
class _HandlerMeta:
    event_name: str
    handler_name: str
    input_type: Any  # raw type hint (may be Union, Annotated, etc.)
    return_type: Any  # raw type hint (may be Union, Annotated, etc.)
    docstring: str | None  # handler function docstring


def _unwrap_annotated(t: Any) -> Any:
    """If t is Annotated[X, ...], return X. Otherwise return t."""
    if get_origin(t) is Annotated:
        return get_args(t)[0]
    return t


def _extract_models(t: Any) -> list[Any]:
    """Unwrap Annotated, then unpack Union args. Filter out NoneType."""
    if t is None:
        return []
    t = _unwrap_annotated(t)
    if get_origin(t) is Union:
        types = list(get_args(t))
    else:
        types = [t]
    return [m for m in types if m is not type(None)]


_PRIMITIVE_SCHEMA: dict[type, dict[str, str]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    dict: {"type": "object"},
}


def _type_to_json_schema(t: Any) -> dict[str, Any]:
    """Return a JSON Schema dict for a type."""
    if isinstance(t, type) and issubclass(t, BaseModel):
        return t.model_json_schema()
    return _PRIMITIVE_SCHEMA.get(t, {})


def _parse_docstring(docstring: str | None) -> tuple[str | None, str | None]:
    """Split docstring into (summary, description).

    First line is summary, remaining stripped lines are description.
    """
    if not docstring:
        return None, None
    lines = docstring.strip().splitlines()
    summary = lines[0].strip() if lines else None
    description = None
    if len(lines) > 1:
        remaining = "\n".join(lines[1:]).strip()
        if remaining:
            description = remaining
    return summary, description


def _to_camel_case(name: str) -> str:
    """Convert snake_case handler name to camelCase."""
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _message_name(t: Any, event_name: str, direction: str, index: int) -> str:
    """Derive a message name for a type.

    BaseModel subclasses use their __name__. Primitives get a generated name.
    """
    if isinstance(t, type) and issubclass(t, BaseModel):
        return t.__name__
    if isinstance(t, type):
        return f"{event_name}_{direction}_{t.__name__}_{index}"
    return f"{event_name}_{direction}_{index}"


def _add_model_to_components(
    model: Any,
    name: str,
    component_schemas: dict[str, Any],
    component_messages: dict[str, Any],
) -> None:
    """Add a model's schema and message to the components dicts."""
    if isinstance(model, type) and issubclass(model, BaseModel):
        component_schemas[name] = model.model_json_schema()
    component_messages[name] = {
        "contentType": "application/json",
        "payload": (
            {"$ref": f"#/components/schemas/{name}"}
            if isinstance(model, type) and issubclass(model, BaseModel)
            else _type_to_json_schema(model)
        ),
    }


def generate_asyncapi_schema(
    handlers: list[_HandlerMeta],
    title: str = "Socket.IO API",
    version: str = "1.0.0",
    description: str | None = None,
) -> dict[str, Any]:
    """Generate an AsyncAPI 3.0.0 specification from handler metadata."""
    channels: dict[str, Any] = {}
    operations: dict[str, Any] = {}
    component_schemas: dict[str, Any] = {}
    component_messages: dict[str, Any] = {}

    for handler in handlers:
        event = handler.event_name
        input_models = _extract_models(handler.input_type)
        output_models = _extract_models(handler.return_type)

        # --- Build input channel messages ---
        channel_messages: dict[str, Any] = {}
        for i, model in enumerate(input_models):
            name = _message_name(model, event, "input", i)
            _add_model_to_components(model, name, component_schemas, component_messages)
            channel_messages[name] = {"$ref": f"#/components/messages/{name}"}

        channels[event] = {"address": event, "messages": channel_messages}

        # --- Build reply channel if there are output models ---
        reply_channel_name = f"{event}Reply"
        if output_models:
            reply_messages: dict[str, Any] = {}
            for i, model in enumerate(output_models):
                name = _message_name(model, event, "output", i)
                _add_model_to_components(
                    model, name, component_schemas, component_messages
                )
                reply_messages[name] = {"$ref": f"#/components/messages/{name}"}

            channels[reply_channel_name] = {
                "address": None,
                "messages": reply_messages,
            }

        # --- Build operation ---
        op_name = _to_camel_case(handler.handler_name)
        summary, op_description = _parse_docstring(handler.docstring)

        operation: dict[str, Any] = {
            "action": "receive",
            "channel": {"$ref": f"#/channels/{event}"},
            "messages": [
                {
                    "$ref": f"#/channels/{event}/messages/"
                    f"{_message_name(m, event, 'input', i)}"
                }
                for i, m in enumerate(input_models)
            ],
        }

        if summary:
            operation["summary"] = summary
        if op_description:
            operation["description"] = op_description

        if output_models:
            operation["reply"] = {
                "channel": {"$ref": f"#/channels/{reply_channel_name}"},
                "messages": [
                    {
                        "$ref": f"#/channels/{reply_channel_name}/messages/"
                        f"{_message_name(m, event, 'output', i)}"
                    }
                    for i, m in enumerate(output_models)
                ],
            }

        operations[op_name] = operation

    # --- Assemble top-level spec ---
    info: dict[str, Any] = {"title": title, "version": version}
    if description:
        info["description"] = description

    return {
        "asyncapi": "3.0.0",
        "info": info,
        "channels": channels,
        "operations": operations,
        "components": {
            "schemas": component_schemas,
            "messages": component_messages,
        },
    }
