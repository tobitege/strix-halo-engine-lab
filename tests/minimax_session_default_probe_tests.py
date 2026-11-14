#!/usr/bin/env python3
"""Verify identical observation anchors on the historical and working source."""
import os, pathlib, shutil, struct, subprocess, sys, tempfile
root=pathlib.Path(__file__).resolve().parents[1]
scratch=root/'build/minimax-audit-p8/observer'
scratch.mkdir(parents=True,exist_ok=True)
tool=root/'tools/minimax/session_default_probe.py'
baseline = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('STRIX_MINIMAX_BASE_REVISION') or '05fedd80'
revisions = ['working']
result = subprocess.run(['git', '-C', str(root), 'cat-file', '-e', baseline + '^{commit}'], capture_output=True)
if result.returncode == 0:
    revisions.insert(0, baseline)
else:
    print('SKIP historical anchors: baseline object unavailable')
with tempfile.TemporaryDirectory(prefix='mm-observer-', dir=scratch) as temp:
    tree=pathlib.Path(temp)
    for revision in revisions:
        relatives=('apps/minimax_rank_main.cpp','apps/minimax_serve_main.cpp',
                   'src/minimax/serve/minimax_control.hpp','src/minimax/serve/minimax_lead.hpp')
        if revision == 'working':
            relatives += ('src/minimax/serve/serve_runner.cpp','src/minimax/serve/serve_runtime.cpp',
                          'src/minimax/serve/serve_runner.hpp','src/serve/session_runtime.cpp')
        for relative in relatives:
            data=(root/relative).read_bytes() if revision=='working' else subprocess.check_output(['git','-C',str(root),'show',revision+':'+relative])
            (tree/relative).parent.mkdir(parents=True, exist_ok=True)
            (tree/relative).write_bytes(data)
        subprocess.run([sys.executable,str(tool),str(tree),revision],check=True)
        cli=(tree/'apps/minimax_rank_main.cpp').read_text()
        serve=(tree/'apps/minimax_serve_main.cpp').read_text()
        control=(tree/'src/minimax/serve/minimax_control.hpp').read_text()
        lead=(tree/'src/minimax/serve/minimax_lead.hpp').read_text()
        extracted = len(serve.splitlines()) <= 5
        if extracted:
            assert cli.count('mm_probe::row(')==2 and 'mm_probe::bytes(eng.rank(), "u32"' in cli
            runtime=(tree/'src/minimax/serve/serve_runtime.cpp').read_text()
            runner=(tree/'src/minimax/serve/serve_runner.hpp').read_text()
            shared=(tree/'src/serve/session_runtime.cpp').read_text()
            control=(tree/'src/minimax/serve/minimax_control.hpp').read_text()
            assert 'startup_bytes(rank_, "startup-send"' in runtime
            assert runtime.index('admit_runtime_pair(') < runtime.index('mm_probe::startup.store(false)')
            assert 'MM_DEFAULT_REQUIRE_SESSION' in runtime
            assert 'mm_probe::row(eng, p.slot, p.rows)' in runner
            assert 'mm_probe::row(eng, row.slot, row.pos + 1, first ? rows.size() : 1, first, !first)' in runner
            assert 'mm_probe::bytes(0, "emitted", ids.data()' in shared
            assert 'if (mm_probe::enabled()) queue.shutdown();' in shared
            assert 'mm_probe::bytes(eng.rank(), "send"' in control
            assert runtime.count('std::make_unique<Borrowed>(*transport, opt.rank)') == 2
        else:
            assert cli.count('mm_probe::row(')==2
        if not extracted:
            assert 'queue.submit(std::move(job))' in serve
            assert serve.count('Lead lead(')==1
            assert serve.index('Lead lead(') < serve.index('queue.submit(std::move(job))')
            if revision=='working':
                assert 'session_runtime ? &session_service : nullptr)' in serve[serve.index('Lead lead('):]
                assert serve.index('ms::SessionService session_service;') < serve.index('Lead lead(')
            assert 'mm_probe::bytes(eng.rank(), "send"' in control
            assert 'mm_probe::bytes(eng.rank(), "receive"' in control
            assert serve.count('mm_probe::startup_bytes(rank_, "startup-send"') == 2
            assert serve.count('mm_probe::startup_bytes(rank_, "startup-receive"') == 2
            assert 'mm_probe::bytes(eng_.rank(), "emitted"' in lead
            assert 'if (mm_probe::enabled()) queue_.shutdown();' in lead
        assert 'PROBE_REVISION' not in (tree/'apps/mm_default_probe.hpp').read_text()
    compiler = os.environ.get('CXX') or shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    msvc = None
    if not compiler:
        cl=shutil.which('cl')
        where=shutil.which('vswhere') or r'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
        if not cl and pathlib.Path(where).is_file():
            install=subprocess.check_output([where,'-latest','-products','*','-property','installationPath'],text=True).strip()
            candidates=sorted(pathlib.Path(install).glob('VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe'))
            if candidates:
                cl=str(candidates[-1])
                vcvars=pathlib.Path(install)/'VC/Auxiliary/Build/vcvars64.bat'
                if vcvars.is_file(): msvc=(cl,str(vcvars))
        elif cl:
            msvc=(cl,None)
        compiler=cl
    assert compiler, 'C++ compiler required for observer witness check'
    is_msvc = bool(msvc) or pathlib.Path(compiler).name.lower() in ('cl','cl.exe')
    def compile_cpp(source, executable, includes):
        if is_msvc:
            args=['/nologo','/std:c++20','/EHsc','/Fo:"'+str(source.with_suffix('.obj'))+'"']+['/I"'+str(p)+'"' for p in includes]+['/Fe:"'+str(executable)+'"','"'+str(source)+'"']
            if msvc and msvc[1]:
                batch=tree/'compile-msvc.bat'
                batch.write_text('@echo off\ncall "'+msvc[1]+'" >nul\nif errorlevel 1 exit /b %errorlevel%\n"'+compiler+'" '+' '.join(args)+'\n')
                subprocess.run(['cmd.exe','/d','/c',str(batch)],check=True,capture_output=True,text=True)
            else: subprocess.run([compiler,*[a.replace('"','') for a in args]],check=True,capture_output=True,text=True)
        else:
            subprocess.run([compiler,'-std=c++20',*[item for p in includes for item in ('-I',str(p))],str(source),'-o',str(executable)],check=True)
    if extracted:
        runtime=(tree/'src/minimax/serve/serve_runtime.cpp').read_text()
        start=runtime.index('class Borrowed final')
        borrowed=runtime[start:runtime.index('\n};',start)+3]
    else:
        borrowed = serve[serve.index('class Borrowed final'):serve.index('\n};', serve.index('class Borrowed final')) + 3]
    harness = r'''
    #include "apps/mm_default_probe.hpp"
    #include "src/minimax/serve/minimax_control.hpp"
    #include <algorithm>
    #include <span>
    namespace strix::dist {
    struct Transport {
        virtual ~Transport() = default;
        virtual void send(std::span<const std::byte>) = 0;
        virtual void receive(std::span<std::byte>) = 0;
        virtual void exchange(std::span<const std::byte>, std::span<std::byte>) = 0;
        virtual std::string name() const = 0;
        virtual void abort(const std::string&) = 0;
    };
    }
    BORROWED
    struct Echo final : strix::dist::Transport {
        void send(std::span<const std::byte>) override {}
        void receive(std::span<std::byte> m) override { std::fill(m.begin(), m.end(), std::byte{}); }
        void exchange(std::span<const std::byte> o, std::span<std::byte> i) override {
            std::copy(o.begin(), o.end(), i.begin());
            // The echoed peer is the other rank of the pair: the lockstep header check rejects a same-rank Control.
            if (i.size() == 512) i[offsetof(strix::minimax::serve::Control, rank)] ^= std::byte{1};
        }
        std::string name() const override { return "echo"; }
        void abort(const std::string&) override {}
    };
    struct Engine {
        Borrowed& transport;
        unsigned rank() const { return 0; }
        strix::minimax::engine::PfProgress prefill_progress(bool = false) const { return {}; }
        void abort(const char*) {}
        void exchange_bytes(const void* out, void* in, std::size_t n) {
    transport.exchange({static_cast<const std::byte*>(out), n}, {static_cast<std::byte*>(in), n});
        }
    };
    int main(int argc, char**) {
        Echo echo; Borrowed transport(echo, 0);
        Engine engine{transport};
        std::array<std::byte, 777> out{}, in{};
        transport.send(std::span(out).first(8));
        transport.receive(std::span(in).first(8));
        (void)strix::minimax::serve::session_hello(engine, argc > 1);
        if (argc > 1) transport.exchange(std::span(out).first(524), std::span(in).first(524));
        mm_probe::startup.store(false);
        transport.exchange(out, in);
        strix::minimax::serve::Control command;
        command.command = 1;
        (void)strix::minimax::serve::exchange(engine, command);
        struct RowEngine {
    std::vector<float> last_logits(std::size_t n) { return std::vector<float>(n*4, 1.25f); }
    void export_rows(unsigned, unsigned, std::uint64_t n, unsigned char* p) { std::fill_n(p, n*76800, 0x5a); }
    unsigned rank() const { return 1; }
        } row_engine;
        mm_probe::row(row_engine, 2, 1);
        const std::uint32_t id=42; mm_probe::bytes(1,"emitted",&id,4);
    }
    '''.replace('BORROWED', borrowed).replace('#include <algorithm>', '#include <algorithm>\n#include <array>\n#include <cstddef>')
    (tree/'startup.cpp').write_text(harness)
    executable = tree/('startup-check.exe' if is_msvc else 'startup-check')
    compile_cpp(tree/'startup.cpp',executable,[root/'include',tree])
    def echoed(sent, received, controls):
        # The echo returns every byte except the rank of each 512-byte Control, which it flips 0 -> 1.
        assert len(sent) == len(received)
        diff = [k for k in range(len(sent)) if sent[k] != received[k]]
        assert len(diff) == controls and all(sent[k] == 0 and received[k] == 1 for k in diff), diff
    def witness(tag, extra):
        prefix = tree/tag
        subprocess.run([str(executable)] + extra, env=dict(os.environ, MM_DEFAULT_PROBE=str(prefix)), check=True)
        sent = pathlib.Path(str(prefix) + '.r0.startup-send').read_bytes()
        echoed(sent, pathlib.Path(str(prefix) + '.r0.startup-receive').read_bytes(), 1)
        ordinary = pathlib.Path(str(prefix) + '.r0.send').read_bytes()
        assert len(ordinary) == 512
        echoed(ordinary, pathlib.Path(str(prefix) + '.r0.receive').read_bytes(), 1)
        assert pathlib.Path(str(prefix) + '.r1.f32').stat().st_size == 16
        assert pathlib.Path(str(prefix) + '.r1.state').stat().st_size == 76800
        assert pathlib.Path(str(prefix) + '.r1.emitted').read_bytes() == struct.pack('=I', 42)
        sizes, at = [], 0
        while at < len(sent):
            size, = struct.unpack_from('=Q', sent, at)
            sizes.append(size); at += 8 + size
        assert at == len(sent)
        return sizes, sent, ordinary
    original_sizes, original, ordinary = witness('ordinary', [])
    extra_sizes, extra, with_sessions = witness('extra', ['vote'])
    assert original_sizes == [8, 512] and extra_sizes == [8, 512, 524]
    assert original != extra, 'startup witness must detect an added non-Control exchange'
    assert ordinary == with_sessions, 'ordinary controls must exclude the session hello vote'
    if extracted:
        runtime=(tree/'src/minimax/serve/serve_runtime.cpp').read_text()
        guard_start=runtime.index('if (mm_probe::enabled() && std::getenv("MM_DEFAULT_REQUIRE_SESSION")')
        guard_end=runtime.index('        if (opt.rank != 0) {',guard_start)
        guard=runtime[guard_start:guard_end].strip()
        refusal=r'''#include <cstdlib>
#include <stdexcept>
#include <string>
struct Opt { unsigned rank; };
int follower=0;
int main() {
bool probe=true; bool session_runtime=false; Opt opt{1};
try { GUARD
    if (opt.rank != 0) ++follower;
} catch (const std::runtime_error& e) {
    if (std::string(e.what()) != "probe requires a live session runtime" || follower != 0) return 3;
    return 0;
}
return 4;
}'''.replace('GUARD',guard).replace('mm_probe::enabled()','probe')
        (tree/'refusal.cpp').write_text(refusal)
        refusal_exe=tree/('refusal.exe' if is_msvc else 'refusal')
        compile_cpp(tree/'refusal.cpp',refusal_exe,[])
        subprocess.run([str(refusal_exe)],env=dict(os.environ, MM_DEFAULT_REQUIRE_SESSION='1'),check=True)
        shared=(tree/'src/serve/session_runtime.cpp').read_text()
        completion='queue.finish(l.taken, std::move(l.stats), std::move(error));'
        finish=shared.index('queue.finish(l.taken, std::move(l.stats), std::move(error),')
        shutdown=shared.index('if (mm_probe::enabled()) queue.shutdown();',finish)
        assert shared[shutdown:shutdown+len('if (mm_probe::enabled()) queue.shutdown();')].startswith('if ')
        lifecycle=r'''#include "apps/mm_default_probe.hpp"
#include <string>
#include <utility>
struct Lane { int taken=7; int stats=9; };
struct Queue { int completed=0; int stopped=0;
void finish(int, int&&, std::string&&) { ++completed; }
void shutdown() { ++stopped; }
};
int main() { Queue queue; Lane l; std::string error; SNIPPET
return queue.completed == 1 && queue.stopped == 1 ? 0 : 5;
}'''.replace('SNIPPET',completion+'\n    '+shared[shutdown:shutdown+len('if (mm_probe::enabled()) queue.shutdown();')])
        (tree/'completion.cpp').write_text(lifecycle)
        completion_exe=tree/('completion.exe' if is_msvc else 'completion')
        compile_cpp(tree/'completion.cpp',completion_exe,[tree])
        subprocess.run([str(completion_exe)],env=dict(os.environ,MM_DEFAULT_PROBE='1'),check=True)
    gate = (root/'tools/minimax/session_cache_gate.sh').read_text()
    assert 'extensions+=(send receive startup-send startup-receive)' in gate
print('PASS observer overlay: historical Lead/follow replay and extracted Runner/session boundaries')
