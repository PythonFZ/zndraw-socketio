# AsyncAPI Schema Generation from Pydantic Models

## Summary

Add AsyncAPI 3.0 schema generation to zndraw-socketio, allowing users to auto-generate event-driven API documentation from their registered Socket.IO handlers and Pydantic models — analogous to FastAPI's OpenAPI/Swagger generation.

## Design Decisions

- **Spec format**: AsyncAPI 3.0 — the event-driven equivalent of OpenAPI
- **Serving**: Schema generation only (`sio.asyncapi_schema() -> dict`). Users serve it themselves.
- **Metadata source**: Pydantic model docstrings + `Field(description=...)` + handler docstrings. No decorator API changes.

## AsyncAPI 3.0 Mapping

| Socket.IO | AsyncAPI 3.0 |
|---|---|
| Event name (`ping`) | Channel with `address: "ping"` |
| `@sio.on(Ping)` handler | Operation with `action: "receive"` |
| Input Pydantic model (`Ping`) | Message payload on the channel |
| Return type (`Pong`) | `reply.messages` on the operation |
| Handler with no return | Operation without `reply` |
| Union return (`Error \| Success`) | Reply channel with multiple messages |
| Union input (`Ping \| AdvancedPing`) | Channel with multiple messages |
| Socket.IO ack (callback) | Reply channel with `address: null` (ephemeral) |

## Type Introspection

```python
hints = get_type_hints(handler, include_extras=True)
sig = inspect.signature(handler)
params = list(sig.parameters.values())

# Skip first str parameter (sid) — detected by type, not name.
# This handles 'sid', 'session_id', etc.
data_param = None
for p in params:
    hint = hints.get(p.name)
    if hint is None:
        continue
    # Unwrap Annotated[T, ...] → T
    inner = get_args(hint)[0] if get_origin(hint) is Annotated else hint
    if inner is not str and _is_payload_type(inner):
        data_param = (p.name, hint)
        break

input_type = data_param[1] if data_param else None

# Unwrap Annotated before checking for Union
def _unwrap(t):
    if get_origin(t) is Annotated:
        t = get_args(t)[0]
    return t

raw = _unwrap(input_type) if input_type else None
input_models = get_args(raw) if get_origin(raw) is Union else [raw] if raw else []

# Output: return annotation (same unwrap logic)
return_type = _unwrap(hints.get("return"))
output_models = get_args(return_type) if get_origin(return_type) is Union else [return_type] if return_type else []
# Filter out NoneType from unions (e.g. Optional[Pong])
output_models = [m for m in output_models if m is not type(None)]
```

All `BaseModel` subclasses get `model_json_schema()` added to `components.schemas`.
Non-BaseModel types (`str`, `dict`, `None`) map to plain JSON Schema types.

## Handler Metadata

```python
@dataclass
class _HandlerMeta:
    event_name: str
    handler_name: str
    input_type: Any             # raw type hint (may be Union, Annotated, etc.)
    return_type: Any            # raw type hint (may be Union, Annotated, etc.)
    docstring: str | None       # handler function docstring
```

Unpacking of `Annotated`, `Union`, etc. happens in `generate_asyncapi_schema()`, not at registration time.

Handler docstrings are split into:
- First line → `operation.summary`
- Remaining lines → `operation.description`

Populated during `@sio.on()` / `@sio.event` registration alongside the existing wrapping logic.

## Public API

```python
def asyncapi_schema(
    self,
    title: str = "Socket.IO API",
    version: str = "1.0.0",
    description: str | None = None,
) -> dict[str, Any]:
    """Generate an AsyncAPI 3.0 specification from registered handlers."""
```

## File Structure

```
src/zndraw_socketio/
├── __init__.py          # add asyncapi_schema re-export (if needed)
├── wrapper.py           # add _handler_registry + _HandlerMeta to wrapper classes
├── asyncapi.py          # NEW — schema generation logic
└── params.py            # unchanged
```

- **`asyncapi.py`**: `_HandlerMeta` dataclass, `generate_asyncapi_schema()` pure function, type introspection helpers
- **`wrapper.py`**: `self._handler_registry: list[_HandlerMeta] = []` on each wrapper, `on()`/`event()` append metadata, new `asyncapi_schema()` method delegates to `generate_asyncapi_schema()`

## Example Output

Given:
```python
@tsio.on(Ping)
async def handle_ping(sid: str, data: Ping) -> Pong:
    """Health check endpoint."""
    return Pong(reply=data.message)
```

Generated spec:
```json
{
  "asyncapi": "3.0.0",
  "info": { "title": "Socket.IO API", "version": "1.0.0" },
  "channels": {
    "ping": {
      "address": "ping",
      "messages": {
        "Ping": { "$ref": "#/components/messages/Ping" }
      }
    },
    "pingReply": {
      "address": null,
      "messages": {
        "Pong": { "$ref": "#/components/messages/Pong" }
      }
    }
  },
  "operations": {
    "handlePing": {
      "action": "receive",
      "summary": "Health check endpoint.",
      "channel": { "$ref": "#/channels/ping" },
      "messages": [{ "$ref": "#/channels/ping/messages/Ping" }],
      "reply": {
        "channel": { "$ref": "#/channels/pingReply" },
        "messages": [{ "$ref": "#/channels/pingReply/messages/Pong" }]
      }
    }
  },
  "components": {
    "schemas": {
      "Ping": {
        "type": "object",
        "properties": { "message": { "type": "string" } },
        "required": ["message"]
      },
      "Pong": {
        "type": "object",
        "properties": { "reply": { "type": "string" } },
        "required": ["reply"]
      }
    },
    "messages": {
      "Ping": {
        "contentType": "application/json",
        "payload": { "$ref": "#/components/schemas/Ping" }
      },
      "Pong": {
        "contentType": "application/json",
        "payload": { "$ref": "#/components/schemas/Pong" }
      }
    }
  }
}
```

## Deferred (YAGNI)

- **`List[Model]` returns** — no usage in codebase; handle when needed.
- **Schema name collisions** — Pydantic's `model_json_schema()` handles `$defs` disambiguation already.

## No New Dependencies

`model_json_schema()` from Pydantic is already available. No external packages needed.
