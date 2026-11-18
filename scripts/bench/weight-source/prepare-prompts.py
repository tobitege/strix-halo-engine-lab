#!/usr/bin/env python3
"""Size distinct benchmark texts using a local tokenizer; server usage remains authoritative."""
import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reserve', type=int, default=128, help='allowance for chat template tokens')
    args = parser.parse_args()
    if not 0 <= args.reserve <= 256:
        parser.error('reserve must be in [0,256]')
    config = json.loads(args.tokenizer.read_text(encoding='utf-8'))
    if isinstance(config.get('model'), str):
        # A GGUF-extracted tokenizer describes its regex and sibling BPE files.
        if not config.get('pre_pattern'):
            parser.error('extracted tokenizer must declare its pre_pattern')
        tokenizer = Tokenizer(models.BPE.from_file(str(args.tokenizer.with_name('vocab.json')),
                                                   str(args.tokenizer.with_name('merges.txt'))))
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(config['pre_pattern']), behavior='isolated'),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        tokenizer.decoder = decoders.ByteLevel()
    else:
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
    text = args.source.read_text(encoding='utf-8')
    source_ids = tokenizer.encode(text, add_special_tokens=False).ids
    if len(source_ids) < 43008:
        parser.error('source must contain at least 43008 plain text tokens')
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
                    scope='plain text counts; check actual server chat usage', prompts={})
    for target in (8192, 32768, 43008):
        manifest['prompts'][str(target)] = []
        for state in ('cold', 'warm'):
            salt = hashlib.sha256(f'{target}/{state}/weight-source-v1'.encode()).hexdigest()
            prefix = f'Benchmark case {salt}. Read the following text and reply briefly.\n'
            budget = target - args.reserve - len(tokenizer.encode(prefix, add_special_tokens=False).ids)
            prompt = prefix + tokenizer.decode(source_ids[:budget], skip_special_tokens=False)
            count = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            if abs(count - (target - args.reserve)) > 8:
                raise ValueError('tokenizer round trip changed the intended size')
            path = args.out / f'pf{target}-{state}.txt'
            path.write_text(prompt, encoding='utf-8')
            manifest['prompts'][str(target)].append(dict(path=str(path), text_tokens=count,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print(f'{path.name}: text_tokens={count} target_chat_tokens={target}')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
