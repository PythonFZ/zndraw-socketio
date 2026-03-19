"""AsyncAPI 3.0.0 schema generation from handler metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from zndraw_socketio.params import Emits


@dataclass
class _HandlerMeta:
    event_name: str
    handler_name: str
    input_type: Any  # raw type hint (may be Union, Annotated, etc.)
    return_type: Any  # raw type hint (may be Union, Annotated, etc.)
    docstring: str | None  # handler function docstring
    emits: list[type[BaseModel]] = field(default_factory=list)  # side-effect events


@dataclass
class _RestEmitterMeta:
    handler_name: str  # function name
    method: str  # HTTP method (GET, POST, PUT, etc.)
    path: str  # route path
    summary: str | None  # endpoint docstring first line
    emits: list[type[BaseModel]] = field(default_factory=list)  # socket models emitted


def _get_event_name(model: type[BaseModel]) -> str:
    """Derive event name from a BaseModel class (local to avoid circular import)."""
    if hasattr(model, "event_name"):
        return model.event_name  # type: ignore[return-value]
    name = model.__name__
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


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


def scan_routes(routes: list[Any]) -> list[_RestEmitterMeta]:
    """Scan FastAPI-style routes for Emits annotations.

    For each route whose endpoint has an ``Annotated[..., Emits(...)]`` parameter,
    returns a ``_RestEmitterMeta`` with the route metadata and emitted models.
    """
    results: list[_RestEmitterMeta] = []
    for route in routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        try:
            hints = get_type_hints(endpoint, include_extras=True)
        except Exception:
            continue

        emitted: list[type[BaseModel]] = []
        for hint in hints.values():
            if get_origin(hint) is not Annotated:
                continue
            for metadata in get_args(hint)[1:]:
                if isinstance(metadata, Emits):
                    emitted.extend(metadata.models)

        if not emitted:
            continue

        methods = getattr(route, "methods", None) or set()
        method = next(iter(sorted(methods)), "GET")
        path = getattr(route, "path", "")
        doc = endpoint.__doc__
        summary, _ = _parse_docstring(doc)

        results.append(
            _RestEmitterMeta(
                handler_name=endpoint.__name__,
                method=method,
                path=path,
                summary=summary,
                emits=emitted,
            )
        )
    return results


def generate_asyncapi_schema(
    handlers: list[_HandlerMeta],
    title: str = "Socket.IO API",
    version: str = "1.0.0",
    description: str | None = None,
    rest_emitters: list[_RestEmitterMeta] | None = None,
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

    # --- Collect emits and build bidirectional x-emits / x-triggered-by ---
    # Map: emit model class → list of receive operation names that emit it
    emit_model_triggers: dict[type, list[str]] = {}
    for handler in handlers:
        receive_op_name = _to_camel_case(handler.handler_name)
        for model in handler.emits:
            emit_model_triggers.setdefault(model, []).append(receive_op_name)

    # Create send channels/operations and annotate receive operations
    for model, trigger_op_names in emit_model_triggers.items():
        emit_event = _get_event_name(model)
        name = model.__name__
        send_op_name = f"send{name}"

        # Add schema + message to components
        _add_model_to_components(model, name, component_schemas, component_messages)

        # Create channel if not already present
        if emit_event not in channels:
            channels[emit_event] = {
                "address": emit_event,
                "messages": {name: {"$ref": f"#/components/messages/{name}"}},
            }
        else:
            # Add message to existing channel if not present
            ch_msgs = channels[emit_event]["messages"]
            if name not in ch_msgs:
                ch_msgs[name] = {"$ref": f"#/components/messages/{name}"}

        # Create send operation with x-triggered-by
        if send_op_name not in operations:
            operations[send_op_name] = {
                "action": "send",
                "channel": {"$ref": f"#/channels/{emit_event}"},
                "messages": [{"$ref": f"#/channels/{emit_event}/messages/{name}"}],
                "x-triggered-by": list(trigger_op_names),
            }
        else:
            # Merge additional triggers into existing send operation
            existing_refs = set(operations[send_op_name].get("x-triggered-by", []))
            for op in trigger_op_names:
                if op not in existing_refs:
                    operations[send_op_name].setdefault("x-triggered-by", []).append(op)

        # Add x-emits to each triggering receive operation
        for op in trigger_op_names:
            if op in operations:
                operations[op].setdefault("x-emits", [])
                if send_op_name not in operations[op]["x-emits"]:
                    operations[op]["x-emits"].append(send_op_name)

    # --- Process REST emitters → x-rest-triggers on send operations ---
    if rest_emitters:
        # Map: emit model class → list of REST trigger dicts
        rest_model_triggers: dict[type, list[dict[str, Any]]] = {}
        for emitter in rest_emitters:
            trigger_info: dict[str, Any] = {
                "operationId": _to_camel_case(emitter.handler_name),
                "method": emitter.method,
                "path": emitter.path,
            }
            if emitter.summary:
                trigger_info["summary"] = emitter.summary
            for model in emitter.emits:
                rest_model_triggers.setdefault(model, []).append(trigger_info)

        for model, triggers in rest_model_triggers.items():
            emit_event = _get_event_name(model)
            name = model.__name__
            send_op_name = f"send{name}"

            # Ensure components exist
            _add_model_to_components(model, name, component_schemas, component_messages)

            # Ensure channel exists
            if emit_event not in channels:
                channels[emit_event] = {
                    "address": emit_event,
                    "messages": {name: {"$ref": f"#/components/messages/{name}"}},
                }
            else:
                ch_msgs = channels[emit_event]["messages"]
                if name not in ch_msgs:
                    ch_msgs[name] = {"$ref": f"#/components/messages/{name}"}

            # Create or update send operation
            if send_op_name not in operations:
                operations[send_op_name] = {
                    "action": "send",
                    "channel": {"$ref": f"#/channels/{emit_event}"},
                    "messages": [{"$ref": f"#/channels/{emit_event}/messages/{name}"}],
                }

            operations[send_op_name].setdefault("x-rest-triggers", []).extend(triggers)

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
