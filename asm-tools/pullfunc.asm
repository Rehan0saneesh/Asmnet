default rel

section .data
         filename db "store.txt", 0
         BUFSIZE equ 1024

section .bss
         filehandle resq 1
         file_buffer resb BUFSIZE
         out_buffer resb BUFSIZE

section .text
         global pullfunc

pullfunc:
         push r12
         push r13

         mov r12, rdi
         mov r13, rsi

         mov rax, 2
         lea rdi, [filename]
         mov rsi, 0
         mov rdx, 0
         syscall

         test rax, rax
         js ext_error
         mov qword [filehandle], rax

         mov rax, 0
         mov rdi, qword [filehandle]
         lea rsi, [file_buffer]
         mov rdx, BUFSIZE
         syscall

         mov rcx, r12
         add rcx, r13
         cmp rax, rcx
         jl ext_error

         mov rax, 3
         mov rdi, qword [filehandle]
         syscall

         lea rsi, [file_buffer]
         add rsi, r12

         lea rdi, [out_buffer]
         mov rcx, r13
         rep movsb

         lea rax, [out_buffer]
         mov rdx, r13

         pop r13
         pop r12
         ret

ext_error:
         mov rax, 60
         mov rdi, 1
         syscall
