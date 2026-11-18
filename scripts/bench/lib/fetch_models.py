"""Download only selected public assets, verifying every file before use."""
import argparse
import fnmatch
import hashlib
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


PREFIX = 'hyukudan/'
TRELLIS = 'DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3'
QWEN = 'Qwen3.8-Flash-Next-GGUF-mirror'
# The distill-g MTP draft head (the engine default), distilled from the mirror's base MTP head.
QWEN_DRAFT = 'Qwen3.8-Flash-Next-MTP-distillG-Q8_0'
UPSTREAM = {
    QWEN: 'unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee',
    QWEN_DRAFT: 'hyukudan/'+QWEN_DRAFT,
    'deepseek-v4.1-flash-gguf-mirror': 'kyuz0/deepseek-v4.1-flash-gguf@20408f69',
    'GLM-5.3-Flash-exl3-mirror': 'turboderp/GLM-5.3-Flash-exl3@2a30229e (4.05bpw)',
    'MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw-mirror': 'Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw@c33e7a66',
    'MiniMax-M3.1-Flash-DFlash2-mirror': 'incoai/MiniMax-M3.1-Flash-DFlash2@ac07a7c1',
    'Qwen3.8-Flash-Next-GSQ-RCO-GGUF-mirror': 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF@ed59f920',
    'Qwen3.8-Flash-Next-GSQ-hybrid-Q2': 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF@ed59f920 plus the project composition recipe',
    TRELLIS: 'hyukudan/'+TRELLIS,
    'DeepSeek-V4.1-Flash-1017-strix-dense-U6': 'hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6',
}
U6 = 'DeepSeek-V4.1-Flash-1017-strix-dense-U6'
# Every mirror is read at a fixed revision: its SHA256SUMS and the selected files come from the same commit.
REVISION = {
    QWEN: 'b056e49fcdf2f3251f956c8a89512397e2d2470d',
    QWEN_DRAFT: '1233b2ca95ed5d6f927a3573c3ce5838e52a33c7',
    'deepseek-v4.1-flash-gguf-mirror': '311b7b64963b8eaa38ef18800c62e8046fd44faf',
    'GLM-5.3-Flash-exl3-mirror': 'f162df61e2bbab8d5c626840f0e5b99c16c5ddc8',
    'MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw-mirror': '3534d6330ea6285fed2d50ac7f81d0eb5a201ac3',
    'Qwen3.8-Flash-Next-GSQ-hybrid-Q2': '39bd5d51828c759fc6296d5dc6652eb2aabe514e',
    TRELLIS: '1cb035dd585e0b6d8b9838f6ebeee1d90c611745',
    U6: '289359b3992d3f7838c5c7cbde9390efa09fd870',
}


def recipe(engine, profile):
    if engine in ('deepseek', 'deepseek-rtx'):
        jobs = [('deepseek-v4.1-flash-gguf-mirror', '', ['DeepSeek-V4.1-Flash-1017-Q2.gguf', 'DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf'])]
        if engine == 'deepseek':
            # The draft vocabularies and keep statistics are published with the U6 overlay, not the trellis packs.
            jobs += [(U6, '', ['c-u6.gguf', 'c-u6.json', 'aux/vocab/*.bin', 'aux/draft-keep-stats.jsonl']),
                     (TRELLIS, '', ['ours-K4/*', 'ours-K5/*', 'mixB.txt'])]
        else:
            jobs += [(U6, 'ds41-draft-vocab', ['aux/vocab/*.bin'])]
        return jobs
    if engine == 'qwen':
        jobs = [(QWEN, '', ['MTP/*.gguf'] + ([] if profile == 'gsq' else ['UD-Q4_K_XL/*.gguf']))]
        # Both profiles draft with distill-g; it lands beside the target. Its SHA256SUMS lists the GGUF and LICENSE.
        jobs += [(QWEN_DRAFT, '', ['mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf'])]
        if profile == 'gsq': jobs += [('Qwen3.8-Flash-Next-GSQ-hybrid-Q2', '', ['hybrid-q2.gguf'])]
        return jobs
    patterns = ['*.safetensors', '*.json', '*.txt', '*.jinja', '*.model']
    if engine == 'glm': return [('GLM-5.3-Flash-exl3-mirror', 'glm', patterns)]
    return [('MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw-mirror', 'minimax', patterns + ['dspark/*'])]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def command(argv, dry=False):
    print(shlex.join(map(str, argv)), flush=True)
    if not dry: subprocess.run(list(map(str, argv)), check=True, timeout=86400)


def fetch(repo, subdir, patterns, destination, cli, dry):
    remote = PREFIX+repo
    upstream = UPSTREAM[repo]
    revision = REVISION[repo]
    target = destination/subdir
    with tempfile.TemporaryDirectory(prefix='bench-download-') as td:
        sums = Path(td)/'SHA256SUMS'
        argv = ['curl', '--fail', '--location', '--connect-timeout', '10', '--max-time', '60', '--retry', '0',
                'https://huggingface.co/'+remote+'/resolve/'+revision+'/SHA256SUMS', '--output', 'SHA256SUMS' if dry else str(sums)]
        try:
            command(argv, dry)
            if dry:
                command([cli, 'download', remote, '--revision', revision, '--local-dir', target, '--include', *patterns, 'LICENSE*', 'README*'], True)
                print('Verify selected files against SHA256SUMS; file not found in repo '+remote+'; try upstream '+upstream)
                return
            entries = {}
            for line in sums.read_text(encoding='utf-8').splitlines():
                m = re.fullmatch(r'([0-9a-f]{64})\s+\*?(.+)', line)
                if not m: raise ValueError('invalid SHA256SUMS in '+remote)
                name = m[2]
                if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or '\\' in name:
                    raise ValueError('unsafe manifest path in '+remote)
                entries[name] = m[1]
            # Wildcards select actual manifest members. Required directory/file groups must exist.
            for pattern in patterns:
                if not any(fnmatch.fnmatch(n, pattern) for n in entries) and pattern not in ('*.txt','*.jinja','*.model'):
                    raise ValueError('file not found in repo '+remote+': '+pattern+'; try upstream '+upstream)
            names = [n for n in entries if any(fnmatch.fnmatch(n, p) for p in patterns+['LICENSE*','README*'])]
            target.mkdir(parents=True, exist_ok=True)
            for name in names:
                path = target/name
                # Reuse only byte-identical in-tree vocabulary packs.
                if repo == U6 and name.startswith('aux/vocab/'):
                    local = Path(__file__).resolve().parents[3]/'data/draft-vocab'/Path(name).name
                    if local.is_file() and sha(local) == entries[name]:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(local, path)
                if not path.is_file() or sha(path) != entries[name]:
                    command([cli, 'download', remote, name, '--revision', revision, '--local-dir', target])
                if not path.is_file() or sha(path) != entries[name]: raise ValueError('SHA256 mismatch: '+name)
            # Preserve attribution and published integrity manifest with downloaded assets.
            shutil.copyfile(sums, target/('SHA256SUMS.'+repo))
            if repo == U6 and any(n.startswith('aux/vocab/') for n in names):
                vocab = target if subdir else destination/'vocab'; vocab.mkdir(exist_ok=True)
                for path in (target/'aux/vocab').glob('*.bin'): shutil.copyfile(path, vocab/path.name)
            if repo == QWEN and any(n.startswith('UD-Q4_K_XL/') for n in names):
                merged = destination/'Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf'
                command(['llama-gguf-split', '--merge', target/sorted(n for n in names if n.startswith('UD-Q4_K_XL/'))[0], merged])
                if sha(merged) != '8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2':
                    raise ValueError('merged Qwen SHA256 differs from published recipe')
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError('file not found or download failed in repo '+remote+'; try upstream '+upstream) from exc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', required=True, choices=['qwen','deepseek','deepseek-pair','deepseek-rtx','glm','glm-pair','minimax','minimax-pair'])
    p.add_argument('--model-dir', required=True, type=Path)
    p.add_argument('--profile', default='default')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    engine = a.engine.replace('-pair','')
    cli = next((name for name in ('hf','huggingface-cli') if shutil.which(name)), None)
    if cli is None and not a.dry_run: raise ValueError('install hf or huggingface-cli')
    cli = cli or 'hf'
    for job in recipe(engine, a.profile): fetch(*job, a.model_dir, cli, a.dry_run)
    if engine == 'qwen':
        if a.profile != 'gsq':
            if a.dry_run: command(['llama-gguf-split','--merge',a.model_dir/'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',a.model_dir/'Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf'],True)
        print('The mirror MTP/*.gguf is the base head; the default distill-g head is mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf (SHA-256 5516d3ce407783a424315b6a2a0d8b06341665bb3541b98e5db0fe944add4c3c). Do not rename base to distill-g.')
    if engine == 'deepseek-rtx': print('Supply a valid MODEL.receipt.json and exported STATE.bin separately.')
    return 0


if __name__ == '__main__':
    try: sys.exit(main())
    except (ValueError, OSError) as exc:
        print('fetch-models: '+str(exc), file=sys.stderr)
        sys.exit(1)
