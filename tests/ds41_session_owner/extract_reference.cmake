if(CHECK_FAILURE)
    execute_process(COMMAND "${BINARY}" --fail-sentinel RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 10)
    if(NOT rc EQUAL 1 OR NOT out STREQUAL "" OR NOT err STREQUAL "DS41 owner FAIL: deliberate active-check sentinel\n")
        message(FATAL_ERROR "sentinel rc=${rc} out=${out} err=${err}")
    endif()
    return()
endif()
execute_process(COMMAND "${PYTHON}" -c [=[
import pathlib,sys,hashlib,json
s=pathlib.Path(sys.argv[1]).read_bytes(); o=pathlib.Path(sys.argv[2]);o.mkdir(parents=True,exist_ok=True)
anchors={
 'args':(b'struct Args :',b'// Fill only compatible'),
 'control':(b'enum class Command :',b'// The drafter\'s device bytes'),
 'outcome':(b'struct PrefillOutcome {',b'// The row-split prefill'),
 'transfer':(b'void send_prompt(',b'bool serve_warm_on()'),
 'session':(b'std::optional<SessionCheckpoint> session_operation(',b'// The device prefill of a rank, created'),
 'records':(b'struct Running {',b'// --import-frames PORT (streaming protocol)'),
 'owner':(b'    std::vector<std::uint32_t> pending_commits;  // rank 0',b'// --prefill-rows auto (the pair default):'),
}
m={}
start=b'    std::function<void(serve::SessionRuntime&)> native_turn;'
end=b'    // DS41 request ownership binding ends.'
ownership=s[s.index(start):s.index(end)] if start in s else b''
(o/'ownership.inc').write_bytes(ownership)
m['ownership']={'sha256':hashlib.sha256(ownership).hexdigest(),'bytes':len(ownership)}
for name,(a,z) in anchors.items():
 assert s.count(a)==s.count(z)==1, 'nonunique extraction anchor '+name
 b=s[s.index(a):s.index(z)];(o/(name+'.inc')).write_bytes(b)
 m[name]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
m['source']={'sha256':hashlib.sha256(s).hexdigest(),'bytes':len(s)}
(o/'extraction.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
]=] "${SOURCE}" "${OUTPUT}" RESULT_VARIABLE rc ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "Exact DS41 loop extraction failed: ${err}")
endif()
