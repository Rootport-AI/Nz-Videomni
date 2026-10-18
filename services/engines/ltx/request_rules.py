"""The per-family numeric request rules, and LTX's own table (2.3 and 2.5 share it).

WHY THIS EXISTS. ``GenerateRequest`` (api/models.py) is validated when the body
arrives, before anything knows which base model will run it. The rules that
are true of EVERY engine (shapes, the cross-field checks of IC-LoRA / NAG /
outpaint / inpaint / Alpha Gen, the schema-wide ranges) stay there. The rules
that are a fact about ONE engine family — the frame grid, the spatial
multiple, the fps, the step count and the shifts — live in a table each
family's adapter declares as ``REQUEST_RULES``, and are applied once the
active family is known: ``services.engines.validate_for_family``, called from
api/generate.py right after ``reject_unsupported`` (and from
api/generate_alpha.py; Docs/MULTI_ENGINE_DESIGN.md §7).

The same table is what ``GET /models`` publishes as a base model's
``constraints`` (:meth:`RequestRules.constraints`), so the screens read the
rule from the one place the server enforces it.

ONE PLACE FOR ``None``. ``num_inference_steps`` (and the two shifts) default to
``None`` on the schema, meaning "this family's default". :meth:`RequestRules.fill_defaults`
writes the family default back into the request before it is checked, so every
later reader (the mock, the job log line, the adapters, metadata.json, the mp4
recipe) sees a number (an LTX job records ``8``).

LTX'S MESSAGES ARE THE ONES ITS CLIENTS ALREADY PARSE: "width must be a
multiple of 64", "num_frames must be 8n+1", "distilled pipeline requires
num_inference_steps=8" ..., in a 422 shaped exactly like pydantic's
(``api.errors.request_validation_failed``).

This module imports nothing from api.models at module scope, because
api.models imports the schema-wide bounds below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from api.models import GenerateRequest

# --------------------------------------------------------------------------- #
# schema-wide bounds (GenerateRequest's Field reads these)
# --------------------------------------------------------------------------- #
#
# The widest range ANY family accepts. They are LTX's own numbers (LTX is the
# widest family on all four), declared here so the Field bounds and the LTX
# table below are one number each rather than two copies.

#: Smallest ``num_frames`` the schema accepts (LTX: 8*1+1).
NUM_FRAMES_MIN = 9
#: Largest ``num_frames`` the schema accepts — 20 s at 24 fps (8*60+1). MiniMax H3's
#: largest 17n+5 count happens to be the same number (17*28+5).
NUM_FRAMES_MAX = 481
#: ``frame_rate`` range the schema accepts.
FRAME_RATE_MIN = 1.0
FRAME_RATE_MAX = 60.0


@dataclass(frozen=True)
class SamplingRange:
    """One sampling knob a family exposes (a slider on the screens)."""

    min: float
    max: float
    step: float
    default: float

    def as_dict(self) -> dict:
        return {"min": self.min, "max": self.max, "step": self.step, "default": self.default}


@dataclass(frozen=True)
class RequestRules:
    """One family's numeric rules for a single ``POST /generate``.

    ``frames_step`` / ``frames_offset``: ``num_frames`` must be
    ``frames_step * n + frames_offset``, within ``frames_min``..``frames_max``.
    ``sampling`` is ``None`` when the family exposes no sampling knob (LTX:
    the distilled schedule is fixed), otherwise the knobs by request field
    name; ``default_steps`` is what an omitted ``num_inference_steps`` becomes.
    ``extra`` is a family-specific check that does not fit the table (LTX's
    distilled pipeline), returning the message of the first violation.
    """

    frames_step: int
    frames_offset: int
    frames_min: int
    frames_max: int
    dimension_multiple: int
    frame_rate_min: float
    frame_rate_max: float
    default_steps: int
    sampling: dict[str, SamplingRange] | None = None
    extra: Callable[["GenerateRequest"], str | None] | None = field(default=None, compare=False)

    # ------------------------------------------------------------ publishing

    def constraints(self) -> dict:
        """The ``constraints`` block ``GET /models`` publishes for a base model."""
        return {
            "num_frames": {
                "step": self.frames_step,
                "offset": self.frames_offset,
                "min": self.frames_min,
                "max": self.frames_max,
            },
            "dimension_multiple": self.dimension_multiple,
            "frame_rate": {"min": self.frame_rate_min, "max": self.frame_rate_max},
            "sampling": (
                None
                if self.sampling is None
                else {name: knob.as_dict() for name, knob in self.sampling.items()}
            ),
        }

    # ------------------------------------------------------------ enforcing

    def fill_defaults(self, request: "GenerateRequest") -> None:
        """Write this family's defaults over every ``None`` the rules own."""
        if request.num_inference_steps is None:
            request.num_inference_steps = self.default_steps
        for name, knob in (self.sampling or {}).items():
            if getattr(request, name) is None:
                default = knob.default
                setattr(request, name, int(default) if name == "num_inference_steps" else float(default))

    def violation(self, request: "GenerateRequest") -> str | None:
        """The message of the first rule ``request`` breaks, or ``None``.

        First offender wins (as in the adapters' reject tables), in a fixed
        order: width, height, the frame grid, the frame range, the frame rate,
        the sampling knobs, then the family's own extra check.
        """
        m = self.dimension_multiple
        if request.width % m != 0:
            return f"width must be a multiple of {m}"
        if request.height % m != 0:
            return f"height must be a multiple of {m}"
        if (request.num_frames - self.frames_offset) % self.frames_step != 0:
            return f"num_frames must be {self.frames_step}n+{self.frames_offset}"
        if not self.frames_min <= request.num_frames <= self.frames_max:
            return f"num_frames must be between {self.frames_min} and {self.frames_max}"
        if not self.frame_rate_min <= request.frame_rate <= self.frame_rate_max:
            if self.frame_rate_min == self.frame_rate_max:
                return f"frame_rate must be {self.frame_rate_min:g}"
            return (
                f"frame_rate must be between {self.frame_rate_min:g} "
                f"and {self.frame_rate_max:g}"
            )
        for name, knob in (self.sampling or {}).items():
            value = getattr(request, name)
            if value is not None and not knob.min <= value <= knob.max:
                return f"{name} must be between {knob.min:g} and {knob.max:g}"
        if self.extra is not None:
            return self.extra(request)
        return None


# --------------------------------------------------------------------------- #
# LTX (2.3 and 2.5)
# --------------------------------------------------------------------------- #

#: The distilled schedule's fixed step count (and LTX's default for an omitted
#: ``num_inference_steps``).
LTX_DISTILLED_STEPS = 8


def _ltx_distilled(request: "GenerateRequest") -> str | None:
    """The distilled pipeline runs exactly 8 steps without CFG."""
    if request.pipeline == "distilled":
        if request.num_inference_steps != LTX_DISTILLED_STEPS:
            return f"distilled pipeline requires num_inference_steps={LTX_DISTILLED_STEPS}"
        if request.guidance_scale != 1.0:
            return "distilled pipeline requires guidance_scale=1.0"
    return None


#: LTX's rules. 64 is the two-stage distilled pipeline's spatial multiple
#: (stage 1 at half size through a 32px VAE); a non-64 display size is
#: ``crop_output``'s job. 8n+1 is the causal VAE's frame grid.
LTX_REQUEST_RULES = RequestRules(
    frames_step=8,
    frames_offset=1,
    frames_min=NUM_FRAMES_MIN,
    frames_max=NUM_FRAMES_MAX,
    dimension_multiple=64,
    frame_rate_min=FRAME_RATE_MIN,
    frame_rate_max=FRAME_RATE_MAX,
    default_steps=LTX_DISTILLED_STEPS,
    sampling=None,
    extra=_ltx_distilled,
)
