section .data
	rows1 db "Enter Mat1 Rows: ", 0
	rows1len equ $ - rows1
	cols1 db "Enter Mat1 Cols: ", 0
	cols1len equ $ - cols1
        rows2 db "Enter Mat2 Rows: ", 0
        rows2len equ $ - rows2
        cols2 db "Enter Mat2 Cols: ", 0
        cols2len equ $ - cols2
        matrix db "Enter value: ", 0
        matrix_len equ $ - matrix
	total_cells1 equ rows1 * cols1
	total_cells2 equ rows2 * cols2

section .bss
	matrix1 resb 100
	matrix2 resb 100
	rows1 resb 1
	cols1 resb 1
	rows2 resb 1
	cols2 resb 1
	temp_buf resb 4

section .text
	global _start

_start:
;Mat1:
	mov rax, 1
	mov rdi, 1
	mov rsi, rows1
	mov rdx, rows1len
	syscall

	mov rax, 0
	mov rdi, 0
	mov rsi, rows1
	mov rdx, 2
	syscall
	sub byte [rows1], "0"

	mov rax, 1
	mov rax, 1
	mov rsi, cols1
	mov rdx, cols1len
	syscall

	mov rax, 0
	mov rdi, 0
	mov rsi, cols1
	mov rdx, 2
	sub byte [cols1], "0"

	movzx rax, byte [rows1]
	movzx rcx, byte [cols1]
	imul rax, rcx
	mov r8, rax
	xor rbx, rbx

.convert_loop1:
        movzx rdx, byte [rsi]
        inc rsi

        cmp rdx, 10
        je .mult
        cmp rdx, 0
        je .mult

        cmp rdx, '0'
        jb .convert_loop1
        cmp rdx, '9'
        ja .convert_loop1

        sub rdx, '0'
        imul rax, rdx
        add r11, rdx
        jmp .convert_loop1

.Mat2:

        mov rax, 1
        mov rdi, 1
        mov rsi, rows2
        mov rdx, rows2len
        syscall

        mov rax, 0
        mov rdi, 0
        mov rsi, rows2
        mov rdx, 2
        syscall
        sub byte [rows2], "0"

        mov rax, 1
        mov rax, 1
        mov rsi, cols2
        mov rdx, cols2len
        syscall

        mov rax, 0
        mov rdi, 0
        mov rsi, cols2
        mov rdx, 2
        sub byte [cols2], "0"

        movzx rax, byte [rows2]
        movzx rcx, byte [cols2]
        imul rax, rcx
        mov r9, rax
        xor rbx, rbx

.convert_loop2:
	movzx rdx, byte [rsi]
	inc rsi

	cmp rdx, 10
	je .mult
	cmp rdx, 0
	je .mult

	cmp rdx, '0'
	jb .convert_loop2
	cmp rdx, '9'
	ja .convert_loop2

	sub rdx, '0'
	imul rax, rdx
	add r10, rdx
	jmp .convert_loop2

