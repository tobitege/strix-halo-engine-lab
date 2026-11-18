"""Owned-process Windows job; no process-name enumeration or device discovery."""
import ctypes
from ctypes import wintypes as w
import subprocess
import time


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class BASIC_LIMITS(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                ('PerJobUserTimeLimit', ctypes.c_longlong), ('LimitFlags', w.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', w.DWORD), ('Affinity', ctypes.c_size_t),
                ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]


class EXTENDED_LIMITS(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', BASIC_LIMITS), ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class STARTUPINFO(ctypes.Structure):
    _fields_ = [('cb', w.DWORD), ('lpReserved', w.LPWSTR), ('lpDesktop', w.LPWSTR),
                ('lpTitle', w.LPWSTR), ('dwX', w.DWORD), ('dwY', w.DWORD),
                ('dwXSize', w.DWORD), ('dwYSize', w.DWORD), ('dwXCountChars', w.DWORD),
                ('dwYCountChars', w.DWORD), ('dwFillAttribute', w.DWORD),
                ('dwFlags', w.DWORD), ('wShowWindow', w.WORD), ('cbReserved2', w.WORD),
                ('lpReserved2', ctypes.POINTER(w.BYTE)), ('hStdInput', w.HANDLE),
                ('hStdOutput', w.HANDLE), ('hStdError', w.HANDLE)]


class PROCESSINFO(ctypes.Structure):
    _fields_ = [('hProcess', w.HANDLE), ('hThread', w.HANDLE),
                ('dwProcessId', w.DWORD), ('dwThreadId', w.DWORD)]


class WindowsJob:
    def __init__(self):
        self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            'SetInformationJobObject': ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            'AssignProcessToJobObject': ([w.HANDLE, w.HANDLE], w.BOOL),
            'TerminateJobObject': ([w.HANDLE, w.UINT], w.BOOL),
            'CloseHandle': ([w.HANDLE], w.BOOL),
            'CreateProcessW': ([w.LPCWSTR, w.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, w.BOOL,
                                w.DWORD, ctypes.c_void_p, w.LPCWSTR,
                                ctypes.POINTER(STARTUPINFO), ctypes.POINTER(PROCESSINFO)], w.BOOL),
            'ResumeThread': ([w.HANDLE], w.DWORD),
            'WaitForSingleObject': ([w.HANDLE, w.DWORD], w.DWORD),
            'GetExitCodeProcess': ([w.HANDLE, ctypes.POINTER(w.DWORD)], w.BOOL),
            'TerminateProcess': ([w.HANDLE, w.UINT], w.BOOL),
            'GetStdHandle': ([w.DWORD], w.HANDLE),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.k, name)
            fn.argtypes, fn.restype = args, result
        self.handle = self.k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = EXTENDED_LIMITS()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE; no breakaway
        if not self.k.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        self.processes = []

    def start(self, command):
        si, pi = STARTUPINFO(), PROCESSINFO()
        si.cb = ctypes.sizeof(si)
        si.dwFlags = 0x100  # STARTF_USESTDHANDLES, including redirected harness logs
        si.hStdInput = self.k.GetStdHandle(w.DWORD(-10).value)
        si.hStdOutput = self.k.GetStdHandle(w.DWORD(-11).value)
        si.hStdError = self.k.GetStdHandle(w.DWORD(-12).value)
        flags = 0x4 | subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW  # CREATE_SUSPENDED
        if not self.k.CreateProcessW(None, ctypes.create_unicode_buffer(subprocess.list2cmdline(command)),
                                     None, None, True, flags, None, None, ctypes.byref(si), ctypes.byref(pi)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            # Assign BEFORE any Python/validation/child code can execute. All
            # descendants inherit this job; no breakaway flag is ever accepted.
            if not self.k.AssignProcessToJobObject(self.handle, pi.hProcess):
                raise ctypes.WinError(ctypes.get_last_error())
            if self.k.ResumeThread(pi.hThread) == 0xffffffff:
                raise ctypes.WinError(ctypes.get_last_error())
            self.processes.append(pi.hProcess)
            return pi.hProcess, pi.dwProcessId
        except BaseException:
            self.k.TerminateProcess(pi.hProcess, 4)
            self.k.WaitForSingleObject(pi.hProcess, 1000)
            self.k.CloseHandle(pi.hProcess)
            raise
        finally:
            self.k.CloseHandle(pi.hThread)

    def wait(self, process, seconds):
        status = self.k.WaitForSingleObject(process, max(0, min(0xfffffffe, int(seconds * 1000))))
        if status == 0x102:
            raise TimeoutError('absolute job deadline')
        if status != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        code = w.DWORD()
        if not self.k.GetExitCodeProcess(process, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value

    def close(self):
        if self.handle:
            # Both explicit deadline termination and closing the sole job handle
            # kill the complete tree, even if the worker is stuck in an IO call.
            self.k.TerminateJobObject(self.handle, 4)
            self.k.CloseHandle(self.handle)
            self.handle = None
        end = time.monotonic() + 15
        for process in getattr(self, 'processes', []):
            self.k.WaitForSingleObject(process, max(0, int((end-time.monotonic())*1000)))
            self.k.CloseHandle(process)
        self.processes = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
