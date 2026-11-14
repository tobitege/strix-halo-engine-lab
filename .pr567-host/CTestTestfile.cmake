# CMake generated Testfile for 
# Source directory: D:/tmp/fix-pr567/tests/pr567_host
# Build directory: D:/tmp/fix-pr567/.pr567-host
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test([=[strix-row-layout-tests]=] "D:/tmp/fix-pr567/.pr567-host/Debug/strix-row-layout-tests.exe")
  set_tests_properties([=[strix-row-layout-tests]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/tmp/fix-pr567/tests/pr567_host/CMakeLists.txt;27;add_test;D:/tmp/fix-pr567/tests/pr567_host/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test([=[strix-row-layout-tests]=] "D:/tmp/fix-pr567/.pr567-host/Release/strix-row-layout-tests.exe")
  set_tests_properties([=[strix-row-layout-tests]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/tmp/fix-pr567/tests/pr567_host/CMakeLists.txt;27;add_test;D:/tmp/fix-pr567/tests/pr567_host/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test([=[strix-row-layout-tests]=] "D:/tmp/fix-pr567/.pr567-host/MinSizeRel/strix-row-layout-tests.exe")
  set_tests_properties([=[strix-row-layout-tests]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/tmp/fix-pr567/tests/pr567_host/CMakeLists.txt;27;add_test;D:/tmp/fix-pr567/tests/pr567_host/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test([=[strix-row-layout-tests]=] "D:/tmp/fix-pr567/.pr567-host/RelWithDebInfo/strix-row-layout-tests.exe")
  set_tests_properties([=[strix-row-layout-tests]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/tmp/fix-pr567/tests/pr567_host/CMakeLists.txt;27;add_test;D:/tmp/fix-pr567/tests/pr567_host/CMakeLists.txt;0;")
else()
  add_test([=[strix-row-layout-tests]=] NOT_AVAILABLE)
endif()
