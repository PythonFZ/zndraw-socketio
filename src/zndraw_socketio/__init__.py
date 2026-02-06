from .wrapper import (
    AsyncClientWrapper as AsyncClientWrapper,
)
from .wrapper import (
    AsyncServerWrapper as AsyncServerWrapper,
)
from .wrapper import (
    AsyncSimpleClientWrapper as AsyncSimpleClientWrapper,
)
from .wrapper import (
    EventContext as EventContext,
)
from .wrapper import (
    SimpleClientWrapper as SimpleClientWrapper,
)
from .wrapper import (
    SioRequest as SioRequest,
)
from .wrapper import (
    SyncClientWrapper as SyncClientWrapper,
)
from .wrapper import (
    SyncServerWrapper as SyncServerWrapper,
)
from .wrapper import (
    get_event_name as get_event_name,
)
from .wrapper import (
    wrap as wrap,
)

try:
    from fastapi import Depends as Depends
except ImportError:
    from zndraw_socketio.params import Depends as Depends
