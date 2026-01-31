# ZnDraw SocketIO
This package provides an opinionated typed interface to the python-socketio library using pydantic models.

```python
from zndraw_socketio import Event
from pydantic import BaseModel
import socketio

sio = socketio.AsyncClient() # can be server as well
```

## Emit Pattern
```python
class Ping(Event):
    message: str

message = Ping(message="Hello, World!")
await message.emit(sio, **kwargs) # kwargs are passed to socketio's emit method
```

## Call / RPC Pattern
```python
class Pong(BaseModel):
    reply: str

class Ping(Event[Pong]):
    message: str

message = Ping(message="Hello, World!")
response: Pong = await message.call(sio, **kwargs)  # kwargs are passed to socketio's call method
print(response.reply)
```

## Listen Pattern
```python
class Pong(BaseModel):
    reply: str

class Ping(Event[Pong]):
    message: str

# Server-side usage
@Ping.on(server)
async def handle_ping(sid: str, data: Ping) -> Pong:
    print(f"User {sid} sent {data.message}")
    return Pong(reply="Pong!")

# Client-side usage
@Ping.on(client)
async def handle_ping(data: Ping) -> Pong:
    print(f"Server sent {data.message}")
    return Pong(reply="Pong!")
```

## Namespace Support
```python
class Ping(Event):
    message: str

@Ping.on(sio, namespace="/chat")
async def handle_ping(data: Ping) -> Pong:
    print(f"Received ping: {data.message}")
    return Pong(reply="Pong!")

message = Ping(message="Hello, Chat!")
await message.emit(sio, namespace="/chat")
```

## Class-based Namespaces
```python
class Pong(BaseModel):
    reply: str

class Ping(Event[Pong]):
    message: str


class MyCustomNamespace(socketio.Namespace):
    def on_connect(self, sid, environ):
        pass

    def on_disconnect(self, sid, reason):
        pass
    
    @Ping.on
    def on_ping(self, sid, data: Ping):
        Ping(message="my_response").emit(self, **kwargs) # kwargs are passed to socketio's emit method
        return Pong(reply="Pong from class-based namespace!")

    @Ping.on # will raise ValueError("`ping` event name does not match `not_ping` eventhandler")!
    async def on_not_ping(self, sid, data: Ping):

sio.register_namespace(MyCustomNamespace('/test'))
```


## Event Names
By default, the event name is the class name in snake_case. You can customize it by setting the `event_name` attribute.

```python
class CustomEvent(Event):
    ...

assert CustomEvent.event_name == "custom_event"
```

you can override it like this:

```python
from typing import ClassVar

class CustomEvent(Event):
    event_name: ClassVar[str] = "my_custom_event"
```