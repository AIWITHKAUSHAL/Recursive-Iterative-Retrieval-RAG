"""Deterministic contract tests; these are not live model evaluations."""
import unittest
import json
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError
from rag import Chunk, Euri, Pipeline, normalize


class EuriTransportTests(unittest.TestCase):
    def test_assessment_accepts_missing_lists_null_and_formatted_ids(self):
        context = [{'chunk': {'id': 'H001'}}]
        for missing, sufficient, expected in ((['eligibility', 'approver'], False, 'eligibility; approver'),
                                               ([], True, ''), (None, True, '')):
            response = json.dumps({'sufficient': sufficient, 'reason': 'Evidence check',
                                   'missing': missing, 'evidence_ids': [' [H001] ', 'H001']})
            with self.subTest(missing=missing), patch.object(Euri, 'complete', return_value=response) as call:
                result = Euri('test-key').assess('question', context)
                self.assertEqual(result['missing'], expected)
                self.assertEqual(result['evidence_ids'], ['H001'])
                self.assertEqual(call.call_count, 1)

    def test_assessment_retries_bad_schema_json_and_unknown_evidence(self):
        good = json.dumps({'sufficient': True, 'reason': 'Supported', 'missing': '',
                           'evidence_ids': ['H001']})
        invalid = ['not JSON', '[]', '{"sufficient":"false"}',
                   good.replace('H001', 'UNKNOWN'), good.replace('"missing": ""', '"missing": "gap"')]
        for bad in invalid:
            with self.subTest(bad=bad), patch.object(Euri, 'complete', side_effect=[bad, good]) as call:
                self.assertTrue(Euri('test-key').assess('question', [{'chunk': {'id': 'H001'}}])['sufficient'])
                self.assertEqual(call.call_count, 2)
                self.assertIn('format_correction', call.call_args.args[1])

    def test_assessment_persistent_failure_is_bounded_and_redacted(self):
        bad = '{"sufficient":true,"reason":"private document text","missing":"","evidence_ids":["SECRET_ID"]}'
        with patch.object(Euri, 'complete', return_value=bad) as call:
            with self.assertRaisesRegex(ValueError, 'outside the supplied passages') as error:
                Euri('test-key').assess('question', [{'chunk': {'id': 'H001'}}])
            self.assertEqual(call.call_count, 2)
            self.assertNotIn('SECRET_ID', str(error.exception))
            self.assertNotIn('private document text', str(error.exception))

    def test_assessment_transport_error_is_not_schema_retry(self):
        with patch.object(Euri, 'complete', side_effect=RuntimeError('Connection failed')) as call:
            with self.assertRaises(RuntimeError):
                Euri('test-key').assess('question', [])
            self.assertEqual(call.call_count, 1)

    def test_decomposition_contract_and_original_question(self):
        provider = Euri('test-key')
        with patch.object(provider, 'complete', return_value='{"questions": [" probation? "]}') as complete:
            self.assertEqual(provider.decompose('original', 'child', [], {'missing': 'eligibility'},
                                                ['previous']), ['probation?'])
            payload = complete.call_args.args[1]
            self.assertEqual(payload['original_question'], 'original')
            self.assertEqual(payload['subquestion'], 'child')
            self.assertEqual(payload['previous_queries'], ['previous'])
        with patch.object(provider, 'complete', return_value='{"questions": "invalid"}'):
            with self.assertRaises(ValueError):
                provider.decompose('original', 'child', [], {}, [])

    def test_explicit_client_headers_and_authentication(self):
        with patch('rag.urlopen') as open_request:
            open_request.return_value.__enter__.return_value = BytesIO(b'{"data": []}')
            Euri('test-key').request('/embeddings', {'model': 'test-model'})
            request = open_request.call_args.args[0]
            self.assertEqual(request.get_header('User-agent'), 'Retrace-RAG/0.1')
            self.assertEqual(request.get_header('Accept'), 'application/json')
            self.assertEqual(request.get_header('Authorization'), 'Bearer test-key')

    def test_gateway_and_api_errors_are_distinct_and_redacted(self):
        for code, body, expected in (
            (403, b'error code: 1010', 'gateway blocked'),
            (403, b'test-key forbidden', 'denied access'),
            (401, b'test-key', 'authentication'),
            (429, b'test-key', 'quota limit'),
        ):
            with self.subTest(code=code, body=body):
                error = HTTPError('https://example.com', code, 'error', {}, BytesIO(body))
                with patch('rag.urlopen', side_effect=error):
                    with self.assertRaises(RuntimeError) as caught:
                        Euri('test-key').request('/embeddings', {'model': 'test-model'})
                self.assertIn(expected, str(caught.exception))
                self.assertIn('/embeddings', str(caught.exception))
                self.assertNotIn('test-key', str(caught.exception))


class FakeProvider:
    def __init__(self, enough=True, repeated=False):
        self.enough, self.repeated = enough, repeated
        self.generated = None

    def embed(self, texts):
        return [[0, 1] if 'probation' in t else [1, 0] for t in texts]

    def assess(self, question, context):
        ids = [h['chunk']['id'] for h in context]
        sufficient = self.enough and len(ids) == 2
        return {'sufficient': sufficient, 'reason': 'Need both policies',
                'missing': '' if sufficient else 'probation eligibility', 'evidence_ids': ids}

    def rewrite(self, question, context, assessment, previous_queries):
        return previous_queries[0] if self.repeated else 'probation'

    def generate(self, question, context, sufficient):
        self.generated = (question, context, sufficient)
        return 'Ineligible during probation [H002].' if sufficient else 'Insufficient evidence.'


class PipelineTests(unittest.TestCase):
    def pipeline(self, **kwargs):
        return Pipeline(FakeProvider(**kwargs), [Chunk('H001', 'policy', 'travel'),
                                                Chunk('H002', 'policy', 'probation')])

    def test_retry_recovers_missing_evidence(self):
        pipeline = self.pipeline()
        result = pipeline.run('Lisbon eligibility?', max_retries=3)
        self.assertEqual([a['query'] for a in result['attempts']], ['Lisbon eligibility?', 'probation'])
        self.assertFalse(result['attempts'][0]['assessment']['sufficient'])
        self.assertTrue(result['sufficient'])
        self.assertEqual(result['stop_reason'], 'sufficient_evidence')
        self.assertEqual(len(result['final_context']), 2)
        self.assertEqual(pipeline.provider.generated[0], 'Lisbon eligibility?')

    def test_zero_retries_is_one_shot(self):
        result = self.pipeline().run('Lisbon?', max_retries=0)
        self.assertEqual(len(result['attempts']), 1)
        self.assertFalse(result['sufficient'])
        self.assertEqual(result['stop_reason'], 'retry_limit')
        self.assertEqual(result['final_assessment'], result['attempts'][-1]['assessment'])

    def test_repeated_rewrite_stops(self):
        result = self.pipeline(repeated=True).run('Lisbon?')
        self.assertEqual(result['stop_reason'], 'repeated_query')
        self.assertEqual(len(result['attempts']), 1)

    def test_retry_limit_and_deduplication(self):
        result = self.pipeline(enough=False).run('Lisbon?', max_retries=1)
        self.assertEqual(len(result['attempts']), 2)
        self.assertEqual(result['stop_reason'], 'retry_limit')
        self.assertFalse(result['sufficient'])

    def test_already_sufficient_stops_immediately(self):
        result = self.pipeline().run('Lisbon?', top_k=2)
        self.assertEqual(len(result['attempts']), 1)
        self.assertTrue(result['sufficient'])

    def test_invalid_settings(self):
        for kwargs in ({'max_retries': -1}, {'max_retries': 6}, {'top_k': 0}):
            with self.assertRaises(ValueError):
                self.pipeline().run('question', **kwargs)

    def test_invalid_vectors(self):
        for vector in ([0, 0], [float('nan')], [True]):
            with self.assertRaises(ValueError):
                normalize(vector)

    def test_unknown_citation_rejected(self):
        pipeline = self.pipeline()
        pipeline.provider.generate = lambda *args: 'Claim [UNKNOWN].'
        with self.assertRaises(ValueError):
            pipeline.run('Lisbon?')


class RecursiveProvider(FakeProvider):
    """Fake branching retrieval; the root needs both policies."""
    def __init__(self, plans=None, enough=True):
        super().__init__(enough=enough)
        self.plans = plans if plans is not None else {'Lisbon?': ['travel detail', 'probation']}
        self.assessed = []

    def assess(self, question, context):
        self.assessed.append(question)
        result = super().assess(question, context)
        if question == 'travel detail':
            result.update(sufficient=True, missing='')
        return result

    def decompose(self, question, subquestion, context, assessment, previous_queries):
        return self.plans.get(subquestion, [])


class RecursivePipelineTests(unittest.TestCase):
    def pipeline(self, **kwargs):
        return Pipeline(RecursiveProvider(**kwargs),
                        [Chunk('H001', 'policy', 'travel'), Chunk('H002', 'policy', 'probation')])

    def test_sibling_evidence_merges_and_local_success_is_not_global_success(self):
        pipeline = self.pipeline()
        result = pipeline.run_recursive('Lisbon?')
        self.assertEqual([a['query'] for a in result['attempts']],
                         ['Lisbon?', 'travel detail', 'probation'])
        self.assertEqual([n['parent_id'] for n in result['nodes']], [None, 1, 1])
        self.assertEqual(result['nodes'][1]['status'], 'subquestion_resolved')
        self.assertTrue(result['sufficient'])
        self.assertEqual(result['stop_reason'], 'sufficient_evidence')
        self.assertEqual(len(result['final_context']), 2)
        self.assertEqual(pipeline.provider.generated[0], 'Lisbon?')
        self.assertEqual(pipeline.provider.assessed.count('Lisbon?'), 3)

    def test_recursive_grandchild(self):
        result = self.pipeline(plans={'Lisbon?': ['eligibility'],
                                      'eligibility': ['probation']}).run_recursive('Lisbon?')
        self.assertEqual([n['depth'] for n in result['nodes']], [0, 1, 2])
        self.assertEqual([n['parent_id'] for n in result['nodes']], [None, 1, 2])
        self.assertTrue(result['sufficient'])

    def test_depth_limit_produces_partial_answer(self):
        result = self.pipeline().run_recursive('Lisbon?', max_depth=0)
        self.assertEqual(len(result['attempts']), 1)
        self.assertEqual(result['stop_reason'], 'depth_limit')
        self.assertEqual(result['final_assessment'], result['attempts'][-1]['assessment'])
        self.assertFalse(result['sufficient'])

    def test_global_search_budget(self):
        result = self.pipeline().run_recursive('Lisbon?', max_searches=2)
        self.assertEqual(len(result['attempts']), 2)
        self.assertEqual(result['stop_reason'], 'search_limit')
        self.assertFalse(result['sufficient'])

    def test_cycle_is_skipped_but_other_branches_continue(self):
        result = self.pipeline(plans={'Lisbon?': ['LISBON?', 'probation']}).run_recursive('Lisbon?')
        self.assertEqual(result['nodes'][1]['status'], 'repeated_query')
        self.assertEqual(len(result['attempts']), 2)
        self.assertTrue(result['sufficient'])

    def test_no_decomposition_and_repeated_only(self):
        for plans, stop in (({}, 'decomposition_exhausted'),
                            ({'Lisbon?': ['LISBON?']}, 'repeated_query')):
            result = self.pipeline(plans=plans).run_recursive('Lisbon?')
            self.assertEqual(result['stop_reason'], stop)
            self.assertFalse(result['sufficient'])

    def test_immediate_success(self):
        result = self.pipeline().run_recursive('Lisbon?', top_k=2)
        self.assertEqual(len(result['attempts']), 1)
        self.assertTrue(result['sufficient'])

    def test_invalid_recursive_settings(self):
        for kwargs in ({'max_depth': -1}, {'max_depth': 5}, {'max_depth': True},
                       {'max_searches': 0}, {'max_searches': 21}, {'top_k': 11}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.pipeline().run_recursive('Lisbon?', **kwargs)

    def test_invalid_decomposition_rejected(self):
        for children in (None, 'probation', [''], ['x']*4, [123], ['x'*2001]):
            with self.subTest(children=children), self.assertRaises(ValueError):
                self.pipeline(plans={'Lisbon?': children}).run_recursive('Lisbon?')

    def test_unknown_citation_and_provider_errors_propagate(self):
        pipeline = self.pipeline()
        pipeline.provider.generate = lambda *args: 'Claim [UNKNOWN].'
        with self.assertRaises(ValueError):
            pipeline.run_recursive('Lisbon?')
        pipeline.provider.decompose = lambda *args: (_ for _ in ()).throw(RuntimeError('Provider failed'))
        with self.assertRaisesRegex(RuntimeError, 'Provider failed'):
            pipeline.run_recursive('Lisbon?')


if __name__ == '__main__':
    unittest.main()
