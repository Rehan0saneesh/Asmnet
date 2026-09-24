section .data
	filename db "store.txt", 0
	BUFSIZE equ 1024

section .bss
	filehandle resq 1
	file_buffer resb BUFSIZE

section .text
	global pullfunc

pullfunc:
	push r12
	push r13

	mov r12, rdi
	mov r13, rsi

	mov rax, 2
	mov rdi, filename
	mov rsi, 0
	mov rdx, 0
	syscall

	test rax, rax
	js ext_error
	mov [filehandle], rax

	mov rax, 0
	mov rdi, [filehandle]
	mov rsi, file_buffer
	mov rdx, BUFSIZE
	syscall

	mov offset_idx, r12
	mov pull_len, r13
	cmp rax, offset_idx + pull_len
	jl ext_error

	mov rax, 3
	mov rdi, [filehandle]
	syscall

	mov rsi, file_buffer
	add rsi, offset_idx

	mov rdi, out_buffer
	mov rcs, pull_len
	rep movsb

	mov rax, out_buffer
	mov rdx, pull_len

	ret

ext_error:
	mov rax, 60
	mov rdi, 1
	syscall
