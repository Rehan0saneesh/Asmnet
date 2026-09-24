section .data
        offset_idx dq 20
        pull_len   dq 4

section .bss
        test_out resq 1
        test_len resq 1

section .text
        extern pullfunc
        global _start

_start:
        mov rdi, [offset_idx]
        mov rsi, [pull_len]

        call pullfunc
        mov [test_out], rax
        mov [test_len], rdx

        mov rax, 1
        mov rdi, 1
        mov rsi, [test_out]
        mov rdx, [test_len]
        syscall

        mov eax, 60
        xor edi, edi
        syscall

