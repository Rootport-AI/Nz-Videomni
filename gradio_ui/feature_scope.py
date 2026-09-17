"""Server feature name -> Gradio control: the ONE place a declared feature
limitation turns into a closed piece of this UI.

Mirror of the WebView2 frontend's ``webui/src/shell/featureScope.ts``: the
server names FEATURES (``prune_vaed``, ``two_stage_hq``, ...) because that is
what a request field is, while a GUI has controls. This module is the
translation, and it is the only place in :mod:`gradio_ui` allowed to know a
server feature name — every caller receives a finished set of control ids.

Hiding a control is only half the job: a Gradio component keeps SENDING its
value while invisible, so a choice made on a base model that supports the
feature would keep riding along after a switch and 422 every job. Hence
:data:`RESET_VALUES`: a hidden control is also written back to the server
default.

A feature with no control here is simply absent from the table rather than
listed with an empty tuple — ``keep_resident_embeddings`` (no Gradio control)
and ``two_stage_hq`` (the quality radio already falls back to distilled) are
names a live backend publishes today that this UI has nothing to close.
Unknown names are ignored for the same reason the frontend ignores
them: a build of this UI is older than the server it talks to more often than
the reverse, so an unrecognised name is the ordinary case, not an error.
"""

from __future__ import annotations

from typing import Iterable

#: Feature name -> the control id(s) it closes. Control ids are defined here
#: and nowhere else; ``ui.py`` maps them to components.
FEATURE_UI: dict[str, tuple[str, ...]] = {
    "prune_vaed": ("accel_vae",),
}

#: The value a hidden control is reset to — the server's own default for the
#: field it feeds, so an invisible control cannot keep a rejected value on the
#: wire. Every id named in :data:`FEATURE_UI` must appear here.
RESET_VALUES: dict[str, str] = {
    "accel_vae": "default",
}

#: Every gated control id, in :data:`FEATURE_UI` order and deduplicated. This
#: is the order ``ui.py`` appends the gated components to its refresh output
#: list in, so the two stay in step.
GATED_CONTROLS: tuple[str, ...] = tuple(dict.fromkeys(
    control for controls in FEATURE_UI.values() for control in controls))


def hidden_controls(unsupported: Iterable[str]) -> frozenset[str]:
    """The control ids the active base model's ``unsupported_features`` hides.

    Unknown feature names are ignored, so the ordinary case (a base model whose
    limitations this UI has no control for, or a backend that publishes no list
    at all) answers with an empty set and no special-casing anywhere."""
    hidden: set[str] = set()
    for name in unsupported or ():
        hidden.update(FEATURE_UI.get(name, ()))
    return frozenset(hidden)
