"""Source closure audit; no CUDA or native result is inferred."""
from pathlib import Path
import argparse, json, subprocess, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_audit import legacy_path, moved_paths  # CUDA backend move: old parent paths and renames

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--git-dir',required=True);ap.add_argument('--source',type=Path,required=True)
    a=ap.parse_args();root=a.source;parent='0f8b7a083cace7bb61f2fe2cf303bc8c69f2dfe3'
    def old(p):
        return moved_paths(subprocess.check_output(['git','--git-dir='+a.git_dir,'show',parent+':'+legacy_path(p)]).decode('utf-8').replace('\r\n','\n'))
    results={}
    for p in ['src/cuda/models/qwen/qwen_mtp_draft.cu','include/strix/cuda/qwen_mtp_draft.hpp','src/cuda/models/qwen/qwen_moe.cu',
              'src/cuda/models/qwen/qwen_qsa.cu','src/cuda/models/qwen/qwen_plemtp.cu','include/strix/cuda/qwen_route_bf16.hpp',
              'apps/cuda_qwen_serve_main.cu','apps/cuda_qwen_prefix_cache.hpp','include/strix/prefill_identity.hpp',
              'include/strix/qwen_profile_table.hpp','src/cuda/models/qwen/qwen_handoff.cpp']:
        results['unchanged:'+p]=(root/p).read_text()==old(p)
    driver=(root/'src/cuda/models/qwen/qwen_prefill_driver.cu').read_text();prior=old('src/cuda/models/qwen/qwen_prefill_driver.cu')
    def math(s):
        s=s[s.index('Draft Prefill::draft(SessionId'):];s=s[s.index('    const auto mw ='):]
        return s[:s.index('\nstd::vector<Draft> Prefill::draft_cohort')]
    results['real_proposer_math_unchanged']=math(driver)==math(prior)
    results['ordinary_guard_retained']='&& !config.diagnostic_bf16_mtp)' in driver and 'if (!impl_->cfg.diagnostic_bf16_mtp)' in driver
    results['no_global_disable']=all(x not in driver for x in ['setenv(','_putenv','m.use_graphs = false'])
    results['actual_live_pair_export']='out.mtp_rows.data(),s.mtp_kv,live*2048' in driver and 's.mtp_index,live*256' in driver
    results['retained_pool_export']='"canonical retained pool"' in driver
    fixture=(root/'tests/cuda/qwen_bf16_mtp_e2e.cu').read_text()
    results['actual_before_reference']=fixture.index('const auto actual=e.capture_canonical()')<fixture.index('e.run(history,history.size(),drop);qp::CanonicalState kept')
    results['actual_proposer_and_target_api']=all(x in fixture for x in ['e.draft(pending,7,0.7F)','e.verify(rows.data()','e.commit_cohort({keep})','md::observe(r.logits','rows_equal(r.logits,scalar_raw)','same(actual,kept)'])
    results['real_forced_separation']='if (arm=="real")' in fixture and 'actual_proposer=' in fixture
    results['replay_actual_not_repaired']='e.restore_canonical(actual);const auto replay=e.step' in fixture
    cmake=(root/'cmake/QwenCuda.cmake').read_text()
    results['production_link']='target_link_libraries(qwen_bf16_mtp_e2e PRIVATE strix_qwen_cuda)' in cmake
    for name in ['include/strix/cuda/qwen_mtp_diagnostic.hpp','tests/cuda/qwen_bf16_mtp_e2e.cu','src/cuda/models/qwen/qwen_prefill_driver.cu']:
        results['LF:'+name]=b'\r' not in (root/name).read_bytes()
    print(json.dumps({'checks':len(results),'passed':sum(results.values()),'results':results},indent=2))
    return 0 if all(results.values()) else 1
if __name__=='__main__':raise SystemExit(main())
