"""Deterministic contract tests; these are not live model evaluations."""
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError
from rag import Chunk, Euri, Pipeline, normalize


class EuriTransportTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
