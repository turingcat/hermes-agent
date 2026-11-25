"""Zed renders its model and effort pickers only from ``configOptions`` (#143547 ACP-24)."""
import pytest
from acp.schema import ModelInfo, SessionModelState

from acp_adapter.server import HermesACPAgent


@pytest.mark.asyncio
async def test_effort_pick_applies_through_the_reasoning_mutation_and_returns_the_pickers(monkeypatch):
    monkeypatch.setattr('acp_adapter.gateway_session_meta.model_state', lambda info: SessionModelState(
        available_models=[ModelInfo(model_id='openrouter:fixture', name='fixture')], current_model_id='openrouter:fixture'))
    info = {'model': 'fixture', 'provider': 'openrouter', 'reasoning_effort': 'medium'}
    mutations = []

    class RPC:
        async def rpc(self, method, **params):
            if method == 'session.mutate':
                mutations.append((params['operation'], params['payload']))
                info['reasoning_effort'] = params['payload']['effort']
                return {'status': 'applied'}
            if method == 'session.info':
                return dict(info)
            return {'revision': 1, 'execution_generation': 1, 'info': dict(info)}

    agent = HermesACPAgent()
    agent._gateway = RPC()
    agent._snapshots['s'] = {'revision': 1, 'execution_generation': 1}
    reply = await agent.set_config_option(config_id='reasoning_effort', session_id='s', value='high')
    assert mutations == [('reasoning', {'effort': 'high'})]
    pickers = {option.category: option.current_value for option in reply.config_options}
    assert pickers == {'mode': 'ask', 'model': 'openrouter:fixture', 'thought_level': 'high'}


def test_the_effort_picker_offers_default_only_while_no_level_is_set():
    from acp_adapter.config_options import config_options
    from acp_adapter.modes import DEFAULT_MODE

    def effort(level):
        option = config_options(None, {'reasoning_effort': level}, DEFAULT_MODE)[-1]
        return option.current_value, [choice.value for choice in option.options]
    # ``default`` (a reset) reads back as the profile's level: offering it would make the pick jump.
    current, values = effort('medium')
    assert current == 'medium' and 'default' not in values
    current, values = effort('')
    assert current == 'default' and values[0] == 'default'


def test_model_rows_advertise_declared_efforts_and_the_level_they_would_run_at():
    """A client that offers a picker per model reads ``ModelInfo._meta``: the session-level
    ``config_options`` option only describes the model in force, so the other rows need their own."""
    from acp_adapter.config_options import advertise_reasoning
    from acp.schema import ModelInfo, SessionModelState

    def row(model_id, efforts):
        meta = {} if efforts is None else {"hermes": {"reasoning_efforts": efforts}}
        return ModelInfo(model_id=model_id, name=model_id, field_meta=meta)

    state = SessionModelState(
        available_models=[row('declared', ['low', 'high']), row('plain', []), row('legacy', None)],
        current_model_id='declared')
    rows = {model.model_id: model.field_meta
            for model in advertise_reasoning(state, {'reasoning_effort': 'max'}).available_models}

    # Declared levels are authoritative, and the reported level is the session's choice constrained
    # to them — max is not runnable on a route that declares low..high.
    assert rows['declared']['supportsReasoningEffort'] is True
    assert rows['declared']['reasoningEfforts'] == [{'value': 'low', 'label': 'Low'},
                                                    {'value': 'high', 'label': 'High'}]
    assert rows['declared']['reasoningEffort'] == 'high'
    # The ACP metadata the row already carried is kept alongside the advertisement.
    assert rows['declared']['hermes']['reasoning_efforts'] == ['low', 'high']

    # An explicit empty declaration means the route takes no effort at all.
    assert rows['plain']['supportsReasoningEffort'] is False
    assert 'reasoningEfforts' not in rows['plain'] and 'reasoningEffort' not in rows['plain']

    # No declaration keeps the legacy ladder, labels included.
    assert [choice['value'] for choice in rows['legacy']['reasoningEfforts']][:2] == ['none', 'minimal']
    assert rows['legacy']['reasoningEfforts'][-1] == {'value': 'ultra', 'label': 'Ultra'}
    assert rows['legacy']['reasoningEffort'] == 'max'


def test_model_rows_never_invent_a_level_and_an_explicit_none_is_reported_where_declared():
    from acp_adapter.config_options import advertise_reasoning
    from acp.schema import ModelInfo, SessionModelState

    def row(model_id, efforts):
        meta = {} if efforts is None else {"hermes": {"reasoning_efforts": efforts}}
        return ModelInfo(model_id=model_id, name=model_id, field_meta=meta)

    state = SessionModelState(
        available_models=[row('declared', ['low', 'high']), row('offable', ['none', 'low']), row('legacy', None)],
        current_model_id='declared')

    # With no session level set, every row advertises its list and none reports a level. A declared
    # row must not inherit ``constrain_reasoning``'s medium fallback here.
    for info in ({}, {'reasoning_effort': ''}, {'reasoning_effort': 'turbo'}):
        rows = advertise_reasoning(state, info).available_models
        assert all('reasoningEffort' not in model.field_meta for model in rows), info

    # An explicit disable is a level where the route declares it, and clamps upward where it does not.
    rows = {model.model_id: model.field_meta
            for model in advertise_reasoning(state, {'reasoning_effort': 'none'}).available_models}
    assert rows['offable']['reasoningEffort'] == 'none'
    assert rows['declared']['reasoningEffort'] == 'low'
