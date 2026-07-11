"""Custom CSS for the Gradio Blocks UI in ``gradio_ui/ui.py``.

This module only defines the :data:`CUSTOM_CSS` constant. Wiring it in
(``gr.Blocks(css=CUSTOM_CSS)``) is done separately in ``ui.py``; this file
does not import from or modify ``ui.py`` in any way.

Dark/light theme detection
---------------------------
Gradio 6.19's own theme bootstrap script (compiled at
``gradio/templates/frontend/assets/Index-BAOWiMqV.js``, function ``Oe``)
toggles a plain ``dark`` class on ``document.body``::

    d==="dark" ? l.classList.add("dark") : l.classList.remove("dark")

-- there is no ``data-theme`` attribute involved anywhere in Gradio 6's
runtime. ``ui.py``'s own theme switcher (``theme_dd.change(..., js=...)``
around line 656) mirrors this exact convention:

    document.body.classList.toggle('light', v === 'light');
    document.body.classList.toggle('dark', v === 'dark');

So every dark-specific override below is scoped with the ancestor selector
``body.dark ...``, matching both Gradio's built-in toggle and this app's
custom JS. Rules that only use Gradio's CSS custom properties (e.g.
``var(--body-text-color-subdued)``, ``var(--input-background-fill)``) do
NOT need an explicit ``body.dark`` branch, because Gradio's own stylesheet
already redefines those variables when ``.dark`` is present on an ancestor
-- a ``body.dark`` branch here is only added where a fixed brand colour
(not a theme variable) must differ per theme, e.g. the base-url amber/blue
swap. Variable names were confirmed against
``gradio/themes/base.py`` (``body_text_color_subdued``, ``input_background
_fill``, ``font_mono``, ...), which Gradio's theme builder emits as
``--body-text-color-subdued`` / ``--input-background-fill`` / ``--font-mono``
CSS custom properties (kebab-cased, ``--`` prefixed).

Group/Form DOM shape (``.duration-panel``)
--------------------------------------------
Gradio 6.19's ``gr.Group`` renders its children through the compiled
``BaseForm`` component (see
``gradio/templates/frontend/assets/BaseForm-DJxyhkeW.css``)::

    div.svelte-d5xbca {
        border: var(--block-border-width) solid var(--block-border-color);
        border-radius: var(--block-radius);
        background: var(--border-color-primary);
        box-shadow: var(--block-shadow);
    }
    div.svelte-d5xbca .block {
        box-shadow: none !important;
        border-width: 0px !important;
        border-radius: 0 !important;
    }

-- i.e. the Group wrapper itself grows the single card border/background, and
EVERY descendant ``.block`` (each Number/Markdown's own chrome) is
automatically stripped to zero border/shadow/radius. No extra CSS is needed
to "merge" the Frames / Duration / Frame-rate trio into one panel; this is
the same built-in behaviour already relied on for the Clip Chain slot
``gr.Group()``s elsewhere in ``ui.py``. The rules below only style the
Duration heading/value text and center it within its narrower middle
``gr.Column`` (``.duration-col``).

Textbox DOM shape
------------------
Gradio's compiled Textbox component (see
``gradio/templates/frontend/assets/Textbox-B5AH0EoL.css``) renders as
``label > textarea`` (or ``label > input``) inside the wrapping ``.block``
div that receives ``elem_classes``, roughly::

    <div class="block ... negative-greyed">
      <label ...>
        <span data-testid="block-info">Negative prompt</span>
        <textarea ...>...</textarea>
      </label>
    </div>

Selectors below therefore target ``textarea``/``input`` (and ``label``)
nested under the ``elem_classes`` hook, rather than a specific Svelte
scoping hash (e.g. ``svelte-1hguek3``), so they keep working across minor
Gradio point releases.
"""

from __future__ import annotations

CUSTOM_CSS: str = """
/* ---- .negative-greyed ---------------------------------------------------
   Negative prompt textbox is functionally disabled (interactive=False;
   distilled mode runs at CFG=1, so the value has no effect on generation).
   Grey it out so the non-editability reads visually in both themes. */
.negative-greyed textarea,
.negative-greyed input {
    background: var(--input-background-fill) !important;
    filter: grayscale(40%);
    opacity: 0.6;
    color: var(--body-text-color-subdued) !important;
    cursor: not-allowed;
}
.negative-greyed label > span {
    color: var(--body-text-color-subdued);
}
body.dark .negative-greyed textarea,
body.dark .negative-greyed input {
    opacity: 0.5;
}

/* ---- .base-url-box -------------------------------------------------------
   Read-only base_url display. Monospace so the URL is easy to scan/copy,
   plus an accent colour that signals "fixed configuration value, not
   user input": blue on light backgrounds, amber on dark (Gradio itself
   favours warmer accents in dark mode, e.g. its own warning/badge colours). */
.base-url-box textarea,
.base-url-box input {
    font-family: var(--font-mono, ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace);
    font-weight: 600;
    letter-spacing: 0.01em;
    color: #0969da;
    cursor: default;
}
body.dark .base-url-box textarea,
body.dark .base-url-box input {
    color: #d29922;
}

/* ---- .duration-panel / .duration-col / .duration-heading / .duration-line -
   Frames (8n+1) / Duration / Frame rate, fused into one panel by gr.Group
   (see the BaseForm note above -- Gradio strips the child blocks' own chrome).

   Owner feedback (2026-07): the panel's background differed from the other
   input panels, and the middle "Duration" text sat at a different height/size
   than the Frames / Frame-rate field labels. Two fixes:

   1. BACKGROUND -- gr.Group's BaseForm wrapper paints its background with
      ``var(--border-color-primary)`` (the pale separator colour that shows
      through the 1px gaps between fused children). A standalone input block
      instead uses ``var(--block-background-fill)``. Override the panel to the
      latter so it reads as the SAME surface as every other operation panel.
      (Confirmed variable name against gradio/themes/base.py
      ``block_background_fill`` -> ``--block-background-fill``; theme-aware, so
      no explicit body.dark branch is needed -- Gradio redefines it per theme.)

   2. HEADING HEIGHT -- match the "Duration" heading to the Number field labels
      so all three top-row captions share one baseline/size. Gradio's Number
      label is ``font-size:var(--text-sm); margin-top:var(--size-2);
      margin-bottom:var(--size-1); color:var(--body-text-color-subdued)``
      (compiled Textbox/Number label rule). The column is top-aligned
      (flex-start, not centred) and the heading given those same metrics, so its
      top edge and text size line up with "Frames"/"Frame rate" beside it. The
      value readout then sits just below in the accent colour, occupying roughly
      the input row. NOTE (Gradio limitation): the value cannot be pixel-locked
      to the exact vertical centre of the neighbouring <input> boxes without
      brittle svelte-hash selectors + hard-coded input heights, so it is aligned
      to the top of the input row rather than its geometric centre. */
.duration-panel {
    align-items: stretch;
    background: var(--block-background-fill);
}
.duration-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    text-align: center;
    padding: 0 var(--spacing-sm, 4px);
}
/* Match the Number field labels' size/top-spacing/colour so the three top-row
   captions (Frames / Duration / Frame rate) align in height. Kept bold so it
   still reads as the summary column's title. */
.duration-heading,
.duration-heading p {
    font-weight: 700;
    font-size: var(--text-sm);
    letter-spacing: 0.02em;
    color: var(--body-text-color-subdued);
    margin: var(--size-2) 0 var(--size-1) 0;
}
/* Duration readout: the AviUtl2-Bridge accent (blue-violet), sized close to the
   input text so it reads as a value rather than an oversized banner. */
.duration-line,
.duration-line p {
    font-size: 1.1em;
    font-weight: 700;
    color: #5b5fc7;
    margin: 0;
    line-height: 1.3;
}
body.dark .duration-line,
body.dark .duration-line p {
    color: #8b95f6;
}

/* ---- .spill-warning -------------------------------------------------------
   Spill-free frame-count warning. Amber text in the same family as
   Gradio's own warning/alert accents, without a heavy box so it doesn't
   fight the surrounding form layout. */
.spill-warning {
    color: #9a6700;
    font-weight: 500;
}
body.dark .spill-warning {
    color: #d29922;
}
"""
