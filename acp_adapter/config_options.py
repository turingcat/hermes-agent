"""Native ACP model groups and per-model reasoning choices shared by both entry points."""

import json

from acp.schema import (
    SessionConfigOptionSelect, SessionConfigSelectGroup, SessionConfigSelectOption, SessionModelState,
)

from acp_adapter.modes import EDIT_APPROVAL_POLICY_CONFIG_ID, MODES

REASONING_CONFIG_ID = "reasoning_effort"
_ADVERTISED_KEYS = ("supportsReasoningEffort", "reasoningEffort", "reasoningEfforts")
REASONING_LABELS = {"none": "None", "minimal": "Minimal", "low": "Low", "medium": "Medium",
                    "high": "High", "xhigh": "Extra High", "max": "Max", "ultra": "Ultra"}


def current_metadata(models: SessionModelState | None, model_id: str | None = None) -> dict:
    if models is None:
        return {}
    selected = model_id if model_id is not None else models.current_model_id
    model = next((model for model in models.available_models if model.model_id == selected), None)
    return (model.field_meta or {}).get("hermes", {}) if model is not None else {}


def reasoning_choices(models: SessionModelState | None = None) -> tuple[str, ...]:
    """An explicit list is authoritative; absent metadata keeps the existing legacy ladder."""
    levels = _advertised_levels(current_metadata(models))
    return ("default", *levels) if levels else ()


def reasoning_effort(config) -> str:
    if not isinstance(config, dict):
        return ""
    return "none" if config.get("enabled") is False else str(config.get("effort") or "")


def constrain_reasoning(config, metadata: dict):
    """Keep a valid session choice, otherwise use the closest supported weaker level."""
    from agent.reasoning_effort import EFFORT_LADDER, clamp_effort
    from hermes_constants import parse_reasoning_effort
    efforts = metadata.get("reasoning_efforts")
    if efforts is None:
        return dict(config) if isinstance(config, dict) else None
    if not efforts:
        return {"enabled": False}
    effort = clamp_effort(reasoning_effort(config) or "medium", efforts)
    if effort not in efforts:
        effort = next((value for value in EFFORT_LADDER if value != "none" and value in efforts), "none")
    return parse_reasoning_effort(effort)


def _advertised_levels(metadata: dict) -> list[str]:
    """The levels a client may offer for this row; absent declarations keep the legacy ladder."""
    from hermes_constants import VALID_REASONING_EFFORTS
    efforts = metadata.get("reasoning_efforts")
    return ["none", *VALID_REASONING_EFFORTS] if efforts is None else list(efforts)


def advertise_reasoning(models: SessionModelState | None, info: dict) -> SessionModelState | None:
    """Publish each row's levels, and the level it would run at, on ``ModelInfo._meta``.

    ``configOptions`` describes the session's current model alone, so a client that renders an
    effort picker per model reads the levels from the rows themselves. ACP leaves that to ``_meta``
    and defines no key for it, so this publishes the bare shape agents already put on the wire
    (``supportsReasoningEffort``, ``reasoningEffort``, ``reasoningEfforts`` as ``{value, label}``):
    a client that reads it can offer effort for a model before a session has run on it, and a client
    that ignores ``_meta`` sees the same session it sees today. The declared ``reasoning_efforts``
    stay the one source of truth — absent declarations fall back to the legacy ladder, and an
    explicit ``[]`` says the route takes no effort at all. A row reports the session's choice
    constrained to what it declares, which is what running on that model would use."""
    if models is None:
        return None
    from hermes_constants import parse_reasoning_effort
    config = parse_reasoning_effort((info or {}).get("reasoning_effort"))
    rows = []
    for row in models.available_models:
        metadata = (row.field_meta or {}).get("hermes") or {}
        levels = _advertised_levels(metadata)
        advertised: dict = {"supportsReasoningEffort": False} if not levels else {
            "supportsReasoningEffort": True,
            "reasoningEfforts": [{"value": level, "label": REASONING_LABELS.get(level, level)}
                                 for level in levels],
        }
        # ``constrain_reasoning`` falls back to ``medium`` when it is handed no level; only ask it
        # once a level is set, so a picker starts unselected instead of inventing one.
        if levels and config is not None:
            level = reasoning_effort(constrain_reasoning(config, metadata))
            if level and level in levels:
                advertised["reasoningEffort"] = level
        kept = {key: value for key, value in (row.field_meta or {}).items() if key not in _ADVERTISED_KEYS}
        rows.append(row.model_copy(update={"field_meta": {**kept, **advertised}}))
    return models.model_copy(update={"available_models": rows})


def _select(config_id, name, category, current, choices):
    return SessionConfigOptionSelect(
        id=config_id, name=name, category=category, type="select", current_value=current,
        options=[SessionConfigSelectOption(value=value, name=label) for value, label in choices])


def _model_option(models: SessionModelState):
    choices = [SessionConfigSelectOption(value=model.model_id, name=model.name, description=model.description)
               for model in models.available_models]
    if not any((model.field_meta or {}).get("hermes", {}).get("group") for model in models.available_models):
        options = choices
    else:
        groups = {}
        for model, choice in zip(models.available_models, choices):
            metadata = (model.field_meta or {}).get("hermes", {})
            provider = metadata.get("provider") or "other"
            label = metadata.get("provider_name") or "Other models"
            group = metadata.get("group")
            key = json.dumps([provider, group], separators=(",", ":"), ensure_ascii=False) if group else provider
            name = f"{label} · {group}" if group else label
            if key not in groups:
                groups[key] = SessionConfigSelectGroup(group=key, name=name, options=[])
            groups[key].options.append(choice)
        options = list(groups.values())
    return SessionConfigOptionSelect(id="model", name="Model", category="model", type="select",
                                    current_value=models.current_model_id, options=options)


def config_options(models: SessionModelState | None, info: dict, mode_id: str) -> list:
    options = [_select(EDIT_APPROVAL_POLICY_CONFIG_ID, "Edits", "mode", MODES[mode_id][0],
                       [(policy, name) for policy, name, _ in MODES.values()])]
    if models is not None:
        options.append(_model_option(models))
    levels = reasoning_choices(models)
    if levels:
        effort = info.get("reasoning_effort") if info.get("reasoning_effort") in levels[1:] else "default"
        choices = levels[1:] if effort != "default" else levels
        options.append(_select(REASONING_CONFIG_ID, "Reasoning effort", "thought_level", effort,
                               [(value, value) for value in choices]))
    return options
