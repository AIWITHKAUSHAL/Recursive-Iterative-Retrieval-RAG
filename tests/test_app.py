"""Headless UI integration using a fake provider; never calls the network."""
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from test_rag import RecursiveProvider


class AppTests(unittest.TestCase):
    def test_incomplete_run_shows_missing_facts_and_stop_guidance(self):
        for mode, limit_label, guidance in (
                ('Iterative', 'Maximum retries', 'retry budget was exhausted'),
                ('Recursive', 'Maximum recursion depth', 'depth limit')):
            with self.subTest(mode=mode):
                provider = RecursiveProvider(enough=False)
                provider.embedding_model = 'fake-embedding'
                provider.chat_model = 'fake-chat'
                with patch('rag.Euri', return_value=provider):
                    app = AppTest.from_file('../app.py', default_timeout=10).run()
                    app.segmented_control[0].set_value(mode).run()
                    next(s for s in app.slider if s.label == limit_label).set_value(0).run()
                    app.button[0].click().run()
                    self.assertFalse(app.exception)
                    self.assertFalse(app.error)
                    self.assertTrue(app.warning)
                    self.assertTrue(any('probation eligibility' in m.value for m in app.markdown))
                    self.assertTrue(any(guidance in i.value for i in app.info))
                    self.assertFalse(app.session_state['result']['sufficient'])

    def test_both_modes_dispatch_and_render(self):
        for mode in ('Iterative', 'Recursive'):
            with self.subTest(mode=mode):
                provider = RecursiveProvider(plans={'Lisbon?': ['probation']})
                provider.embed = lambda texts: [[0, 1] if t == 'probation' or '90-day' in t
                                                else [1, 0] for t in texts]
                provider.embedding_model = 'fake-embedding'
                provider.chat_model = 'fake-chat'
                with patch('rag.Euri', return_value=provider):
                    app = AppTest.from_file('../app.py', default_timeout=10).run()
                    self.assertFalse(app.exception)
                    app.segmented_control[0].set_value(mode).run()
                    self.assertFalse(app.exception)
                    labels = [slider.label for slider in app.slider]
                    self.assertIn('Maximum recursion depth' if mode == 'Recursive'
                                  else 'Maximum retries', labels)
                    app.text_area[0].set_value('Lisbon?')
                    app.button[0].click().run()
                    self.assertFalse(app.exception)
                    self.assertFalse(app.error, [e.value for e in app.error])
                    result = app.session_state['result']
                    self.assertEqual(result['mode'], mode.lower())
                    self.assertTrue(result['sufficient'])
                    if mode == 'Recursive':
                        self.assertTrue(any(e.label == 'Recursive subquestion tree'
                                            for e in app.expander))


if __name__ == '__main__':
    unittest.main()
