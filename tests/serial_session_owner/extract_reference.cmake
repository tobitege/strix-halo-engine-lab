if(CHECK_FAILURE)
    execute_process(COMMAND "${BINARY}" --failure-probe RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 10)
    if(NOT rc EQUAL 1 OR NOT err MATCHES "serial check: argc==0")
        message(FATAL_ERROR "Release assertion sentinel did not fail as intended: ${rc} ${out} ${err}")
    endif()
    return()
endif()
# Python writes slices as bytes. CMake string/file operations can translate line
# endings; the bridge hash pins the actual unchanged production translation.
execute_process(COMMAND "${PYTHON}" -c [=[
import pathlib,hashlib,sys
s=pathlib.Path(sys.argv[1]).read_bytes()
a=b'class MtpBridgeEngine final'; z=b'inline std::atomic<bool>& http_serve_stop_requested()'
assert s.count(a)==s.count(z)==1, 'bridge anchors must be unique'
v=s[s.index(a):s.index(z)]
assert hashlib.sha256(v).hexdigest()=='29c47c286f03fc956ae33998d6eb41272ec605b360d1520752bb5730072f628a', 'bridge changed'
p=pathlib.Path(sys.argv[2]);p.mkdir(parents=True,exist_ok=True)
(p/'serial_bridge.inc').write_bytes(v)
a=b'        // Serial native-control binding begin.'; z=b'        // Serial native-control binding end.'
assert s.count(a)==s.count(z)==1, 'serial binding anchors must be unique'
v=s[s.index(a):s.index(z)]
assert v==b'        // Serial native-control binding begin.\n        serve::SerialSessionOwner owner(queue, engine);\n', 'serial binding changed'
(p/'serial_binding.inc').write_bytes(v)
a=b'inline bool handle_session_route(';z=b'// Serves until SIGINT'
assert s.count(a)==s.count(z)==1
route=s[s.index(a):s.index(z)]
assert hashlib.sha256(route).hexdigest()=='4640b90a45e7c52ddce272853baadb5b9d37d4a0cbc2a506c8304ea453e6c6ba'
h=(pathlib.Path(sys.argv[1]).parent/'hip_session_store.hpp').read_bytes()
a=b'inline void validate_http_session_id(';z=b'inline bool session_prompt_extends'
assert h.count(a)==h.count(z)==1
validate=h[h.index(a):h.index(z)]
assert hashlib.sha256(validate).hexdigest()=='377f9b8d8adb2b22dbd847a0355f0da017c2cda10235f44a9317b0cb72343119'
(p/'serial_routes.inc').write_bytes(validate+route)
assert (p/'serial_bridge.inc').read_bytes().hex()==s[s.index(b'class MtpBridgeEngine final'):s.index(b'inline std::atomic<bool>& http_serve_stop_requested()')].hex()
]=] "${SOURCE}" "${OUTPUT}" RESULT_VARIABLE rc ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "Exact-byte serial extraction failed: ${err}")
endif()
