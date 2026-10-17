"""Record a fresh live demonstration and human-labelled evidence recall."""
import json
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from rag import Chunk, Euri, Pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['iterative', 'recursive'], default='iterative')
    args = parser.parse_args()
    load_dotenv()
    root = Path(__file__).parent
    provider = Euri(os.getenv('EURI_API_KEY', ''))
    chunks = [Chunk(**c) for c in json.loads((root / 'data/handbook.json').read_text())]
    pipeline = Pipeline(provider, chunks)
    question = ('Can an Atlas employee who joined 20 days ago work remotely from Lisbon '
                'for 20 days, and who must approve it?')
    result = (pipeline.run_recursive(question, max_depth=2, max_searches=4, top_k=1)
              if args.mode == 'recursive' else pipeline.run(question, max_retries=3, top_k=1))
    gold = {'H001', 'H002'}
    first = {h['chunk']['id'] for h in result['attempts'][0]['results']}
    final = {h['chunk']['id'] for h in result['final_context']}
    result['evaluation'] = {'required_ids': sorted(gold),
                            'first_evidence_recall': len(first & gold)/len(gold),
                            'final_evidence_recall': len(final & gold)/len(gold),
                            'improved': len(final & gold) > len(first & gold)}
    result['models'] = {'embedding': provider.embedding_model, 'chat': provider.chat_model}
    path = root / ('reports/live-demo-recursive.json' if args.mode == 'recursive'
                   else 'reports/live-demo.json')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['evaluation'], indent=2))
    print(f'Trace saved to {path}')


if __name__ == '__main__':
    main()
