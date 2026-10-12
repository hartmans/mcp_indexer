"""Document plugins: document handles that participate in discovery.

A :class:`DocumentPlugin` is a :class:`DocumentHandle` whose class can be
selected by name through the handler-name registry. A source's discovery
(see :func:`mcp_indexer.plugins.file_source.extension_policy`) maps content
to an interpretation policy carrying a ``handler`` name; the source resolves
that name through :data:`HANDLER_REGISTRY` and constructs the selected
plugin.

Handles that do not participate in discovery (the notmuch probe's threads
and messages) remain plain :class:`DocumentHandle` subclasses and are never
registered here. A plugin that knows its child/delegate type may also
construct it directly with an appropriate policy instead of redispatching
through the registry.
"""
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from .base import DocumentHandle

if TYPE_CHECKING:
    from .base import ContentBridge, DocumentSource

# Global registry of document plugins by handler name, analogous to the
# source registry in mcp_indexer.context. Registration happens when a
# DocumentPlugin subclass is defined (first registration wins; a duplicate
# name does not replace the original class).
HANDLER_REGISTRY: dict[str, type] = {}


class DocumentPlugin(DocumentHandle):
    """A DocumentHandle that participates in discovery.

    Common constructor inputs: ``document_id``, ``source``, ``bridge``,
    ``policy``, ``last_modified``, and ``metadata``.

    - ``policy`` describes the interpretation of this document. It carries
      a string ``handler`` (for example ``"text"``) naming the plugin, plus
      interpretation settings such as boundary regexps. The policy is
      read-only input by convention: construction leaves it unchanged,
      including the ``handler`` key, so one policy can be shared across
      many documents.
    - ``metadata`` describes this document, including its title. The
      constructor takes a shallow copy because the document may modify
      what it retains; :meth:`get_metadata` returns a fresh shallow copy
      on every call, so nested values are not isolated by these copies.
    - ``last_modified`` is the acquisition's authoritative modification
      time. If the incoming metadata also carries ``last_modified``, the
      constructor argument wins: it is the value the source measured for
      this acquisition.
    """

    handler_name: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.handler_name or cls.__name__.lower()
        if name not in HANDLER_REGISTRY:
            HANDLER_REGISTRY[name] = cls

    def __init__(
        self,
        document_id: str,
        source: "DocumentSource",
        bridge: "ContentBridge",
        policy: dict[str, Any],
        last_modified: datetime,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(source, document_id, last_modified)
        self.bridge = bridge
        self.policy = policy
        self.metadata = dict(metadata) if metadata else {}

    async def get_metadata(self) -> dict[str, Any]:
        """Return a fresh shallow copy of this document's metadata.

        The handle's own ``last_modified`` is reported under
        ``last_modified`` and wins over any value carried in incoming
        metadata.
        """
        metadata = dict(self.metadata)
        metadata["last_modified"] = str(self.last_modified)
        return metadata
