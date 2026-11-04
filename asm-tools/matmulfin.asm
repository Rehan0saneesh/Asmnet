section .data
	msg_rows1 db "Enter mat1 rows: ",0
	msg_rows1_len equ $ - msg_rows1
	msg_cols1 db "Enter mat1 cols: ", 0
	msg_cols1_len equ $ - msg_cols1
        msg_rows2 db "Enter mat2 rows: ",0
        msg_rows2_len equ $ - msg_rows2
        msg_cols2 db "Enter mat2 cols: ", 0
        msg_cols2_len equ $ - msg_cols2
	msg_val db "Enter value: ", 0
	msg_val_len equ $ - msg_val
	msg_error db "Error: m1 cols do not equal m2 rows", 10
	msg_err_len equ $ - msg_err
	space db " "
	newline db 10

section .bss
	matrix1 resb 100
	matrix2 resb 100
	matrix3 resd 100
	rows1 resb 1
	cols1 resb 1
        rows2 resb 1
        cols2 resb 1
	temp_buf resb 16

section .text
	global _start

_start:
	mov rax, 1
	mov rdi, 1
	mov rsi, msg_rows1
	mov rdx, msg_rows1_len
	syscall
	call read_digit
	mov [rows1], al

	mov rax, 1
	mov rdi, 1
	mov rsi, msg_cols1
	mov rdx, msg_cols1_len
	syscall
	call read_digit
	mov [cols1], al

	movzx rcx, byte [rows1]
	movzx rax, byte [cols1]
	imul rcx, rax
	lea rbx, [matrix1]

.read_mat1:
	push rcx
	push rbx
	mov rax, 1
	mov rdi, 1
	mov rsi, msg_val
	mov rdx, msg_val_len
	syscall
	call read_number
	pop rbx
	mov [rbx], al
	inc rbx
	pop rcx
	loop .read_mat1

	mov rax, 1
	mov rdi, 1
	mov rsi, mst_rows2
	mov rdx, msg_rows2_len
	syscall
	call read_digit
	mov [rows2], al

	mov rax, 1
	mov rdi, 1
	mov rsi, msg_rows2
	mov rdx, msg_rows2_len
	syscall
	call read_digit
	mov [cols2], al

	mov al, [cols1]
	cmp al, [rows2]
	jne .error

	movzx rcx, byte [rows2]
	movzx rax, byte [cols2]
	imul rcx, rax
	lea rbx, [matrix2]

.read_mat2:
	push rcx
	push rbx
	mov rax, 1
	mov rdi, 1
	mov rsi, msg_val
	mov rdx, msg_val_len
	syscall
	call read_number
	pop rbx
	mov [rbx], al
	inc rbx
	pop rcx
	loop .read_mat2

	xor r8, r8

.loop_i:
	movzx rax, byte [rows1]
	cmp r8, rax
	jge .print_res

	xor r9, r9

.loop_j:
	movzx rax, byte [cols2]
	cmp r9, rax
	jge .next_i

	xor r10, r10
	xor r12, r12

.loop_k:
	movzx rax, byte [cols1]
	cmp r10, rax
	jge .store_res

	mov rax, r8
	movzx rcx, byte [cols1]
	imul rax, rcx
	add rax, r10
	movzx r13, byte [matrix1 + rax]

	mov rax, r10
	movzx rcx, byte [cols2]
	imul rax, rcx
	add rax, r9
	movzx r14, byte [matrix2 + rax]

	imul r13, r14
	add r12, r13

	inc r10
	jmp .loop_k

.store_res:
	mov rax, r8
	movzx rcx, byte [cols2]
	imul rax, rcx
	add rax, r9
	mov dword [matrix3 + rax*4], r12d

	inc r9
	jmp .loop_j

